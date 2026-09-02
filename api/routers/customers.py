"""Customer, explanation, recommendation, segment, and analytics read endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Subquery

from api.database import get_session
from api.models import Customer, Prediction, Recommendation, Segment, Transaction
from api.schemas.serving import (
    CustomerResponse,
    ExplanationResponse,
    ModelInsightsResponse,
    OverviewResponse,
    PriorityCustomerResponse,
    RecommendationResponse,
    RevenuePointResponse,
    SegmentCustomerResponse,
    SegmentSummaryResponse,
)

router = APIRouter(tags=["customers"])


def _latest_predictions() -> Subquery:
    """Return a ranked subquery whose row one is each customer's latest prediction."""
    return select(
        Prediction.customer_id.label("customer_id"),
        Prediction.scored_at.label("scored_at"),
        Prediction.churn_probability.label("churn_probability"),
        Prediction.churn_label.label("churn_label"),
        Prediction.predicted_clv_180d.label("predicted_clv_180d"),
        func.row_number()
        .over(
            partition_by=Prediction.customer_id,
            order_by=(Prediction.scored_at.desc(), Prediction.id.desc()),
        )
        .label("prediction_rank"),
    ).subquery()


def _minmax(values: list[float]) -> list[float]:
    """Min-max normalize values, treating a non-empty constant population as one."""
    if not values:
        return []
    lower, upper = min(values), max(values)
    if upper == lower:
        return [1.0] * len(values)
    return [(value - lower) / (upper - lower) for value in values]


@router.get("/customers/{customer_id}", response_model=CustomerResponse)
def get_customer(
    customer_id: str, session: Annotated[Session, Depends(get_session)]
) -> CustomerResponse:
    """Return one persisted customer summary and current segment."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {customer_id}")
    return CustomerResponse(
        customer_id=customer.customer_id,
        country=customer.country,
        feature_as_of=customer.feature_as_of,
        feature_schema_version=customer.feature_schema_version,
        net_spend=float(customer.net_spend),
        value_tier=customer.value_tier,
        segment_id=customer.segment.segment_id if customer.segment else None,
        segment_name=customer.segment.segment_name if customer.segment else None,
    )


@router.get("/customers/{customer_id}/explanation", response_model=ExplanationResponse)
def get_explanation(
    customer_id: str,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> ExplanationResponse:
    """Return per-customer TreeSHAP drivers without persisting another prediction."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {customer_id}")
    registry = request.app.state.artifacts
    score = registry.score(customer.feature_payload, customer.sequence_payload)
    explanation = registry.explain(customer.feature_payload, score.churn_probability)
    return ExplanationResponse(customer_id=customer.customer_id, **explanation)


@router.get(
    "/customers/{customer_id}/recommendations",
    response_model=list[RecommendationResponse],
)
def get_recommendations(
    customer_id: str,
    session: Annotated[Session, Depends(get_session)],
    limit: int = Query(default=5, ge=1, le=20),
) -> list[RecommendationResponse]:
    """Return persisted ranked item recommendations for one known customer."""
    if session.get(Customer, customer_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {customer_id}")
    rows = session.scalars(
        select(Recommendation)
        .where(Recommendation.customer_id == customer_id)
        .order_by(Recommendation.rank)
        .limit(limit)
    ).all()
    return [
        RecommendationResponse(
            stock_code=row.stock_code,
            rank=row.rank,
            score=float(row.score),
            recommendation_version=row.recommendation_version,
        )
        for row in rows
    ]


@router.get("/segments", response_model=list[SegmentSummaryResponse])
def get_segments(
    session: Annotated[Session, Depends(get_session)],
) -> list[SegmentSummaryResponse]:
    """Return current segment names and customer counts."""
    rows = session.execute(
        select(Segment.segment_id, Segment.segment_name, func.count(Segment.customer_id))
        .group_by(Segment.segment_id, Segment.segment_name)
        .order_by(Segment.segment_id)
    ).all()
    return [
        SegmentSummaryResponse(
            segment_id=int(segment_id),
            segment_name=str(segment_name),
            customer_count=int(count),
        )
        for segment_id, segment_name, count in rows
    ]


@router.get("/segments/{segment_id}", response_model=list[SegmentCustomerResponse])
def get_segment_customers(
    segment_id: int,
    session: Annotated[Session, Depends(get_session)],
    country: str | None = None,
    value_tier: str | None = Query(default=None, pattern="^(low|medium|high|unclassified)$"),
    limit: int = Query(default=5000, ge=1, le=5000),
) -> list[SegmentCustomerResponse]:
    """Return filterable customers assigned to one segment."""
    statement = (
        select(Customer)
        .join(Segment)
        .where(Segment.segment_id == segment_id)
        .order_by(Customer.net_spend.desc(), Customer.customer_id)
        .limit(limit)
    )
    if country is not None:
        statement = statement.where(Customer.country == country)
    if value_tier is not None:
        statement = statement.where(Customer.value_tier == value_tier)
    rows = session.scalars(statement).all()
    return [
        SegmentCustomerResponse(
            customer_id=row.customer_id,
            country=row.country,
            net_spend=float(row.net_spend),
            value_tier=row.value_tier,
        )
        for row in rows
    ]


@router.get("/analytics/revenue", response_model=list[RevenuePointResponse])
def get_revenue(
    session: Annotated[Session, Depends(get_session)],
) -> list[RevenuePointResponse]:
    """Return monthly signed merchandise revenue from loaded serving transactions."""
    dialect = session.bind.dialect.name if session.bind is not None else ""
    period = (
        func.strftime("%Y-%m", Transaction.invoice_timestamp)
        if dialect == "sqlite"
        else func.to_char(Transaction.invoice_timestamp, "YYYY-MM")
    )
    rows = session.execute(
        select(period.label("period"), func.sum(Transaction.quantity * Transaction.price))
        .where(Transaction.is_valid_merchandise.is_(True))
        .group_by(period)
        .order_by(period)
    ).all()
    return [
        RevenuePointResponse(period=str(value), revenue=float(revenue)) for value, revenue in rows
    ]


@router.get("/analytics/overview", response_model=OverviewResponse)
def get_overview(
    session: Annotated[Session, Depends(get_session)],
) -> OverviewResponse:
    """Return concise business metrics for the executive overview."""
    latest = _latest_predictions()
    total_customers = int(session.scalar(select(func.count()).select_from(Customer)) or 0)
    scored_customers = int(
        session.scalar(
            select(func.count()).select_from(latest).where(latest.c.prediction_rank == 1)
        )
        or 0
    )
    high_risk = int(
        session.scalar(
            select(func.count())
            .select_from(latest)
            .where(latest.c.prediction_rank == 1, latest.c.churn_label.is_(True))
        )
        or 0
    )
    total_spend = float(session.scalar(select(func.sum(Customer.net_spend))) or 0.0)
    countries = list(
        session.scalars(
            select(Customer.country)
            .where(Customer.country.is_not(None))
            .distinct()
            .order_by(Customer.country)
        ).all()
    )
    return OverviewResponse(
        total_customers=total_customers,
        scored_customers=scored_customers,
        segment_count=int(
            session.scalar(select(func.count(func.distinct(Segment.segment_id)))) or 0
        ),
        country_count=len(countries),
        total_historical_net_spend=total_spend,
        average_historical_net_spend=total_spend / total_customers if total_customers else 0.0,
        high_risk_customers=high_risk,
        countries=[str(value) for value in countries],
        generated_at=datetime.now(UTC),
    )


@router.get("/analytics/churn-priority", response_model=list[PriorityCustomerResponse])
def get_churn_priority(
    session: Annotated[Session, Depends(get_session)],
    segment_id: int | None = None,
    country: str | None = None,
    value_tier: str | None = Query(default=None, pattern="^(low|medium|high|unclassified)$"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[PriorityCustomerResponse]:
    """Rank latest scores using normalized churn probability times normalized 180-day value."""
    latest = _latest_predictions()
    statement = (
        select(
            Customer.customer_id,
            Customer.country,
            Customer.value_tier,
            Segment.segment_id,
            Segment.segment_name,
            latest.c.scored_at,
            latest.c.churn_probability,
            latest.c.predicted_clv_180d,
        )
        .join(Segment, Segment.customer_id == Customer.customer_id)
        .join(
            latest,
            and_(
                latest.c.customer_id == Customer.customer_id,
                latest.c.prediction_rank == 1,
            ),
        )
    )
    if segment_id is not None:
        statement = statement.where(Segment.segment_id == segment_id)
    if country is not None:
        statement = statement.where(Customer.country == country)
    if value_tier is not None:
        statement = statement.where(Customer.value_tier == value_tier)
    rows = session.execute(statement).all()
    risks = [float(row.churn_probability) for row in rows]
    values = [float(row.predicted_clv_180d) for row in rows]
    normalized_risks = _minmax(risks)
    normalized_values = _minmax(values)
    responses = [
        PriorityCustomerResponse(
            customer_id=str(row.customer_id),
            country=None if row.country is None else str(row.country),
            value_tier=str(row.value_tier),
            segment_id=int(row.segment_id),
            segment_name=str(row.segment_name),
            scored_at=row.scored_at,
            churn_probability=risk,
            predicted_clv_180d=value,
            normalized_churn_probability=normalized_risk,
            normalized_predicted_clv_180d=normalized_value,
            retention_priority=normalized_risk * normalized_value,
        )
        for row, risk, value, normalized_risk, normalized_value in zip(
            rows, risks, values, normalized_risks, normalized_values, strict=True
        )
    ]
    responses.sort(
        key=lambda row: (
            -row.retention_priority,
            -row.churn_probability,
            -row.predicted_clv_180d,
            row.customer_id,
        )
    )
    return responses[:limit]


@router.get("/models/insights", response_model=ModelInsightsResponse)
def get_model_insights(request: Request) -> ModelInsightsResponse:
    """Return safe global explainability and comparison evidence for the dashboard."""
    return ModelInsightsResponse(**request.app.state.artifacts.dashboard_insights())
