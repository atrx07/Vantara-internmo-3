"""STEP 08 dashboard analytics, API boundary, report, and Streamlit view tests."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend.analytics import friendly_feature_name, revenue_forecast
from frontend.api_client import APIClientError, DashboardAPI
from frontend.reports import batch_results_csv, batch_results_pdf

PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeDashboardAPI:
    """In-process dashboard contract fixture used by Streamlit AppTest."""

    def overview(self) -> dict[str, Any]:
        return {
            "total_customers": 4952,
            "scored_customers": 3,
            "segment_count": 2,
            "country_count": 2,
            "total_historical_net_spend": 123456.0,
            "average_historical_net_spend": 24.93,
            "high_risk_customers": 2,
            "countries": ["France", "United Kingdom"],
            "generated_at": "2026-09-02T00:00:00Z",
        }

    def segments(self) -> list[dict[str, Any]]:
        return [
            {"segment_id": 0, "segment_name": "Loyal High Value", "customer_count": 2},
            {"segment_id": 1, "segment_name": "Developing", "customer_count": 1},
        ]

    def churn_priority(self, **_kwargs: object) -> list[dict[str, Any]]:
        return [
            {
                "customer_id": "12346",
                "country": "United Kingdom",
                "value_tier": "high",
                "segment_id": 0,
                "segment_name": "Loyal High Value",
                "scored_at": "2026-09-02T00:00:00Z",
                "churn_probability": 0.8,
                "predicted_clv_180d": 900.0,
                "normalized_churn_probability": 1.0,
                "normalized_predicted_clv_180d": 1.0,
                "retention_priority": 1.0,
            }
        ]

    def segment_customers(self, _segment_id: int, **_kwargs: object) -> list[dict[str, Any]]:
        return [
            {
                "customer_id": "12346",
                "country": "United Kingdom",
                "net_spend": 900.0,
                "value_tier": "high",
            }
        ]

    def model_insights(self) -> dict[str, Any]:
        return {
            "global_churn_importance": [
                {"feature": "recency_days", "mean_absolute_shap": 0.06},
                {"feature": "net_spend", "mean_absolute_shap": 0.03},
            ],
            "churn_comparison": [{"model": "random_forest", "validation_roc_auc": 0.80}],
            "clv_comparison": [{"model": "ridge", "validation_r2": 0.96}],
            "next_category_evaluation": [{"model": "lightgbm", "top_3_accuracy": 0.59}],
            "recommender_evaluation": [{"model": "item_cosine", "hit_rate_at_5": 0.35}],
            "segment_profiles": [
                {
                    "algorithm": "kmeans",
                    "kmeans_segment": 0,
                    "business_label": "Loyal High Value",
                    "recency_days": 52.0,
                    "frequency_orders": 13.9,
                    "net_spend": 5727.0,
                }
            ],
            "segment_pca": [
                {"pca_1": 0.1, "pca_2": 0.3, "kmeans_segment": 0},
                {"pca_1": -0.2, "pca_2": 0.1, "kmeans_segment": 1},
            ],
        }

    def revenue(self) -> list[dict[str, Any]]:
        months = pd.date_range("2024-01-01", periods=24, freq="MS")
        return [
            {"period": month.strftime("%Y-%m"), "revenue": 1000.0 + index * 20}
            for index, month in enumerate(months)
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "freeze_version": "vantara-model-freeze-v1",
            "feature_schema_version": "vantara-churn-features-v1",
            "feature_count": 47,
            "churn": {
                "model": "random_forest",
                "held_out_metrics": {"roc_auc": 0.813, "recall": 0.98},
            },
            "clv": {
                "model": "xgboost_tweedie",
                "version": "vantara-clv-production-v2",
                "development_metrics": {
                    "cv_r2_mean": 0.563,
                    "validation_r2": 0.789,
                },
                "historical_v1_held_out_metrics": {"r2": 0.031},
            },
        }

    def churn_partial_dependence(self) -> str:
        return PNG_1X1

    def score_customer(self, customer_id: str) -> dict[str, Any]:
        return {
            "prediction_id": 1,
            "customer_id": customer_id,
            "as_of_timestamp": "2011-09-10T12:50:00",
            "churn_probability": 0.8,
            "churn_label": True,
            "churn_threshold": 0.2,
            "predicted_clv_180d": 900.0,
            "next_purchase_probability": 0.65,
            "next_category_id": "12",
            "next_category_probability": 0.4,
            "anomaly_score": 0.01,
            "anomaly_flag": False,
            "segment_id": 0,
            "segment_name": "Loyal High Value",
            "model_version": "vantara-model-freeze-v1",
            "persisted": True,
        }

    def customer(self, customer_id: str) -> dict[str, Any]:
        return {"customer_id": customer_id, "country": "United Kingdom"}

    def explanation(self, customer_id: str) -> dict[str, Any]:
        return {
            "customer_id": customer_id,
            "text": "Recent inactivity and order frequency are the strongest model drivers.",
            "positive_drivers": ["recency_days"],
            "negative_drivers": ["frequency_orders"],
        }

    def recommendations(self, _customer_id: str) -> list[dict[str, Any]]:
        return [{"rank": 1, "stock_code": "85123A"}]


def _render_harness(page: str, fake_api: object) -> None:
    import streamlit as st

    from frontend.app import (
        cached_insights,
        cached_metadata,
        cached_overview,
        cached_priority,
        cached_revenue,
        cached_segment_customers,
        cached_segments,
        render_batch_scoring,
        render_churn_priority,
        render_customer_explorer,
        render_executive_overview,
        render_model_insights,
        render_revenue,
        render_segments,
    )
    from frontend.styles import apply_dashboard_style

    for cached in (
        cached_insights,
        cached_metadata,
        cached_overview,
        cached_priority,
        cached_revenue,
        cached_segment_customers,
        cached_segments,
    ):
        cached.clear()
    st.set_page_config(page_title="Vantara Test", layout="wide")
    apply_dashboard_style()
    renderers = {
        "Executive overview": render_executive_overview,
        "Customer segments": render_segments,
        "Churn priorities": render_churn_priority,
        "Customer explorer": render_customer_explorer,
        "Revenue and forecast": render_revenue,
        "Batch scoring": render_batch_scoring,
        "Model insights": render_model_insights,
    }
    renderers[page](fake_api)


def _batch_rows() -> list[dict[str, Any]]:
    return [
        {
            "prediction_id": 17,
            "customer_id": "12346",
            "as_of_timestamp": "2011-09-10T12:50:00",
            "churn_probability": 0.82,
            "churn_label": True,
            "churn_threshold": 0.20,
            "predicted_clv_180d": 925.50,
            "next_purchase_probability": 0.64,
            "next_category_id": "12",
            "next_category_probability": 0.41,
            "anomaly_score": 0.01,
            "anomaly_flag": False,
            "segment_id": 0,
            "segment_name": "Loyal High Value",
            "model_version": "vantara-model-freeze-v1",
            "persisted": True,
        },
        {
            "prediction_id": 18,
            "customer_id": "12347",
            "as_of_timestamp": "2011-09-10T12:50:00",
            "churn_probability": 0.31,
            "churn_label": True,
            "churn_threshold": 0.20,
            "predicted_clv_180d": 210.25,
            "next_purchase_probability": None,
            "next_category_id": "4",
            "next_category_probability": 0.22,
            "anomaly_score": 0.20,
            "anomaly_flag": True,
            "segment_id": 1,
            "segment_name": "Developing",
            "model_version": "vantara-model-freeze-v1",
            "persisted": True,
        },
    ]


def test_dashboard_api_client_handles_json_batch_and_safe_errors() -> None:
    """The frontend client uses only HTTP and converts API failures into user-safe text."""
    observed: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append((request.method, request.url.path))
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/predict/batch"):
            assert "multipart/form-data" in request.headers["content-type"]
            return httpx.Response(200, json=_batch_rows())
        return httpx.Response(404, json={"detail": "Unknown customer_id: SAFE"})

    client = DashboardAPI(
        "http://testserver/api/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.health()["status"] == "ok"
        assert len(client.score_batch("batch.csv", b"customer_id\n12346\n")) == 2
        with pytest.raises(APIClientError, match="Unknown customer_id: SAFE"):
            client.customer("SAFE")
    finally:
        client.close()
    assert observed == [
        ("GET", "/api/v1/health"),
        ("POST", "/api/v1/predict/batch"),
        ("GET", "/api/v1/customers/SAFE"),
    ]


def test_holt_winters_and_short_history_fallbacks() -> None:
    """Revenue forecasting uses seasonal Holt-Winters and deterministic documented fallbacks."""
    full = FakeDashboardAPI().revenue()
    seasonal = revenue_forecast(full, periods=6, seasonal_periods=12)
    assert len(seasonal.forecast) == 6
    assert "12-month seasonality" in seasonal.method
    assert seasonal.forecast["revenue"].ge(0).all()
    trend = revenue_forecast(full[:6], periods=3, seasonal_periods=12)
    assert "damped-trend fallback" in trend.method
    short = revenue_forecast(full[:2], periods=2, seasonal_periods=12)
    assert "Last-observation fallback" in short.method
    assert short.forecast["revenue"].tolist() == [1020.0, 1020.0]
    empty = revenue_forecast([])
    assert empty.historical.empty
    assert "no historical revenue" in empty.method
    assert friendly_feature_name("recency_days") == "Recency Days"


def test_batch_csv_and_pdf_exports_are_auditable() -> None:
    """CSV/PDF downloads retain identifiers, versions, timestamps, scores, and cautions."""
    rows = _batch_rows()
    csv_frame = pd.read_csv(BytesIO(batch_results_csv(rows)))
    assert csv_frame["customer_id"].astype(str).tolist() == ["12346", "12347"]
    assert set(csv_frame["model_version"]) == {"vantara-model-freeze-v1"}
    pdf = batch_results_pdf(rows, generated_at=datetime(2026, 9, 2, tzinfo=UTC))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5000
    assert b"Vantara Batch Scoring Report" in pdf
    assert b"12346" in pdf and b"12347" in pdf
    assert b"vantara-model-freeze-v1" in pdf
    assert b"manual-review" in pdf.lower()
    assert b"not guarantees" in pdf.lower()


@pytest.mark.parametrize(
    ("page", "heading"),
    [
        ("Executive overview", "Customer health, at a glance"),
        ("Customer segments", "Understand every customer group"),
        ("Churn priorities", "Focus retention effort where it matters"),
        ("Customer explorer", "One customer, explained"),
        ("Revenue and forecast", "Revenue history, with a cautious look ahead"),
        ("Batch scoring", "Score a customer batch, safely"),
        ("Model insights", "How the intelligence layer behaves"),
    ],
)
def test_required_streamlit_views_render_without_exceptions(page: str, heading: str) -> None:
    """Every required dashboard view renders through the API-shaped fixture."""
    dashboard = AppTest.from_function(
        _render_harness,
        args=(page, FakeDashboardAPI()),
        default_timeout=15,
    ).run(timeout=15)
    assert not dashboard.exception
    assert any(heading in element.value for element in dashboard.markdown)


def test_customer_search_renders_score_xai_and_recommendations() -> None:
    """Customer search interaction exposes all required individual intelligence surfaces."""
    dashboard = AppTest.from_function(
        _render_harness,
        args=("Customer explorer", FakeDashboardAPI()),
        default_timeout=15,
    ).run(timeout=15)
    dashboard.text_input(key="customer_id").set_value("12346")
    dashboard.button[0].click().run(timeout=15)
    assert not dashboard.exception
    assert len(dashboard.metric) == 4
    assert any("strongest model drivers" in element.value for element in dashboard.markdown)
    assert any("TreeSHAP" in element.value for element in dashboard.caption)


def test_model_insights_identifies_active_v2_and_historical_v1_limit() -> None:
    """The dashboard distinguishes active development evidence from immutable v1 test evidence."""
    dashboard = AppTest.from_function(
        _render_harness,
        args=("Model insights", FakeDashboardAPI()),
        default_timeout=15,
    ).run(timeout=15)
    assert not dashboard.exception
    assert any("vantara-clv-production-v2" in element.value for element in dashboard.success)
    assert any("Historical v1 limitation" in element.value for element in dashboard.warning)


@pytest.mark.parametrize(
    ("page", "country_key", "tier_key"),
    [
        ("Customer segments", "segment_country", "segment_tier"),
        ("Churn priorities", "risk_country", "risk_tier"),
    ],
)
def test_segment_and_priority_filters_rerender_cleanly(
    page: str, country_key: str, tier_key: str
) -> None:
    """Country and value-tier filters remain interactive on both decision views."""
    dashboard = AppTest.from_function(
        _render_harness,
        args=(page, FakeDashboardAPI()),
        default_timeout=15,
    ).run(timeout=15)
    dashboard.selectbox(key=country_key).select("France")
    dashboard.selectbox(key=tier_key).select("high").run(timeout=15)
    assert not dashboard.exception
    assert dashboard.selectbox(key=country_key).value == "France"
    assert dashboard.selectbox(key=tier_key).value == "high"


def test_frontend_has_no_model_database_or_artifact_imports() -> None:
    """The Streamlit package consumes FastAPI and never imports serving internals directly."""
    root = Path(__file__).resolve().parents[1] / "frontend"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    forbidden = ("src.models", "models_artifacts", "sqlalchemy", "api.models", "joblib.load")
    assert all(value not in text for value in forbidden)
