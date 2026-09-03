"""Reusable API orchestration for scoring and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.artifacts import ArtifactRegistry, ScoreResult
from api.models import Customer, Prediction, Segment


def _decimal(value: float, places: str = "0.0000000001") -> Decimal:
    return Decimal(str(value)).quantize(Decimal(places))


def persist_score(
    session: Session,
    customer: Customer,
    score: ScoreResult,
    registry: ArtifactRegistry,
) -> Prediction:
    """Persist one versioned score and refresh the customer's segment assignment."""
    now = datetime.now(UTC)
    prediction = Prediction(
        customer_id=customer.customer_id,
        model_version=registry.serving_version,
        scored_at=now,
        as_of_timestamp=customer.feature_as_of,
        churn_probability=_decimal(score.churn_probability),
        churn_label=score.churn_label,
        churn_threshold=_decimal(score.churn_threshold),
        churn_threshold_version="validation-f2-v1",
        predicted_clv_180d=_decimal(score.predicted_clv_180d, "0.0001"),
        next_purchase_probability=(
            _decimal(score.next_purchase_probability)
            if score.next_purchase_probability is not None
            else None
        ),
        next_category_id=score.next_category_id,
        next_category_probability=_decimal(score.next_category_probability),
        anomaly_score=_decimal(score.anomaly_score),
        anomaly_flag=score.anomaly_flag,
    )
    session.add(prediction)
    segment = session.get(Segment, customer.customer_id)
    if segment is None:
        segment = Segment(
            customer_id=customer.customer_id,
            segment_id=score.segment_id,
            segment_name=score.segment_name,
            model_version="kmeans-production-v1",
            assigned_at=now,
        )
        session.add(segment)
    else:
        segment.segment_id = score.segment_id
        segment.segment_name = score.segment_name
        segment.assigned_at = now
    session.commit()
    session.refresh(prediction)
    return prediction


def score_customer(
    session: Session,
    registry: ArtifactRegistry,
    customer_id: str,
    *,
    persist: bool = True,
) -> tuple[Customer, ScoreResult, Prediction | None]:
    """Fetch, score, and optionally persist one known customer."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise LookupError(f"Unknown customer_id: {customer_id}")
    score = registry.score(customer.feature_payload, customer.sequence_payload)
    prediction = persist_score(session, customer, score, registry) if persist else None
    return customer, score, prediction


def prediction_payload(
    customer: Customer, score: ScoreResult, prediction: Prediction | None
) -> dict[str, Any]:
    """Build the stable HTTP response representation for one score."""
    return {
        "prediction_id": prediction.id if prediction is not None else None,
        "customer_id": customer.customer_id,
        "as_of_timestamp": customer.feature_as_of,
        **score.to_dict(),
        "model_version": prediction.model_version if prediction is not None else None,
        "persisted": prediction is not None,
    }


def prediction_count(session: Session, customer_id: str) -> int:
    """Return persisted prediction count for testable persistence evidence."""
    return len(
        session.scalars(select(Prediction.id).where(Prediction.customer_id == customer_id)).all()
    )
