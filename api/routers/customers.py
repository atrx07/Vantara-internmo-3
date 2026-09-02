"""Customer, explanation, recommendation, segment, and analytics read endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.database import get_session
from api.models import Customer, Recommendation, Segment, Transaction
from api.schemas.serving import (
    CustomerResponse,
    ExplanationResponse,
    RecommendationResponse,
    RevenuePointResponse,
    SegmentCustomerResponse,
    SegmentSummaryResponse,
)

router = APIRouter(tags=["customers"])


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
    limit: int = Query(default=100, ge=1, le=1000),
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
