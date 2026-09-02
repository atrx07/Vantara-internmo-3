"""Validated HTTP contracts for prediction and dashboard-supporting endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Known-customer prediction request."""

    customer_id: str = Field(min_length=1, max_length=64, pattern=r"^[^\s]+$")
    as_of_date: datetime | None = None


class PredictionResponse(BaseModel):
    """Complete persisted multi-model score for one customer."""

    prediction_id: int | None
    customer_id: str
    as_of_timestamp: datetime
    churn_probability: float
    churn_label: bool
    churn_threshold: float
    predicted_clv_180d: float
    next_purchase_probability: float | None
    next_category_id: str
    next_category_probability: float
    anomaly_score: float
    anomaly_flag: bool
    segment_id: int
    segment_name: str
    model_version: str | None
    persisted: bool


class CustomerResponse(BaseModel):
    """Safe customer summary used by the dashboard customer explorer."""

    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    country: str | None
    feature_as_of: datetime
    feature_schema_version: str
    net_spend: float
    value_tier: str
    segment_id: int | None
    segment_name: str | None


class ExplanationResponse(BaseModel):
    """Individual TreeSHAP driver response with deterministic narrative."""

    customer_id: str
    probability: float
    threshold: float
    positive_drivers: list[str]
    negative_drivers: list[str]
    text: str
    method: str
    causal: bool


class RecommendationResponse(BaseModel):
    """One ranked item recommendation."""

    stock_code: str
    rank: int
    score: float
    recommendation_version: str


class SegmentSummaryResponse(BaseModel):
    """Aggregated current segment count."""

    segment_id: int
    segment_name: str
    customer_count: int


class SegmentCustomerResponse(BaseModel):
    """Customer row returned for a selected segment."""

    customer_id: str
    country: str | None
    net_spend: float
    value_tier: str


class RevenuePointResponse(BaseModel):
    """Monthly net-merchandise revenue point."""

    period: str
    revenue: float


class OverviewResponse(BaseModel):
    """Executive customer-health summary derived from persisted serving state."""

    total_customers: int
    scored_customers: int
    segment_count: int
    country_count: int
    total_historical_net_spend: float
    average_historical_net_spend: float
    high_risk_customers: int
    countries: list[str]
    generated_at: datetime


class PriorityCustomerResponse(BaseModel):
    """Latest persisted score ranked by the locked retention-priority formula."""

    customer_id: str
    country: str | None
    value_tier: str
    segment_id: int
    segment_name: str
    scored_at: datetime
    churn_probability: float
    predicted_clv_180d: float
    normalized_churn_probability: float
    normalized_predicted_clv_180d: float
    retention_priority: float


class FeatureImportanceResponse(BaseModel):
    """One global churn importance row."""

    feature: str
    mean_absolute_shap: float


class ModelInsightsResponse(BaseModel):
    """Safe dashboard-facing modeling and explainability evidence."""

    global_churn_importance: list[FeatureImportanceResponse]
    churn_comparison: list[dict[str, Any]]
    clv_comparison: list[dict[str, Any]]
    next_category_evaluation: list[dict[str, Any]]
    recommender_evaluation: list[dict[str, Any]]
    segment_profiles: list[dict[str, Any]]
    segment_pca: list[dict[str, Any]]
