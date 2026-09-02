"""STEP 08 dashboard-supporting FastAPI read-surface tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import create_app
from api.models import Customer, Prediction, Segment
from api.routers.customers import _minmax


def _client(runtime: dict[str, object]) -> TestClient:
    app = create_app(
        database_url=str(runtime["database_url"]),
        project_root=runtime["root"],
        artifact_root=runtime["root"] / "models_artifacts",
    )
    return TestClient(app)


def test_overview_and_model_insights_are_safe_and_complete(
    migrated_runtime: dict[str, object],
) -> None:
    """Executive and model-insight endpoints expose dashboard data without local paths."""
    with _client(migrated_runtime) as client:
        overview = client.get("/api/v1/analytics/overview")
        insights = client.get("/api/v1/models/insights")
        partial_dependence = client.get("/api/v1/models/insights/churn-pdp")
    assert overview.status_code == 200
    assert overview.json()["total_customers"] == 4952
    assert overview.json()["segment_count"] == 8
    assert "United Kingdom" in overview.json()["countries"]
    assert insights.status_code == 200
    payload = insights.json()
    assert payload["global_churn_importance"][0]["feature"] == "recency_days"
    assert len(payload["segment_pca"]) == 1000
    assert all(
        "artifact" not in row and "mlflow_run_id" not in row for row in payload["churn_comparison"]
    )
    assert partial_dependence.status_code == 200
    assert len(partial_dependence.json()["image_base64"]) > 1000


def test_locked_retention_priority_formula_and_latest_score_order(
    migrated_runtime: dict[str, object],
) -> None:
    """Priority is min-max risk times min-max value and uses each customer's latest score."""
    now = datetime.now(UTC)
    identifiers = ["STEP08-A", "STEP08-B", "STEP08-C"]
    risks = [0.1, 0.9, 0.5]
    values = [100.0, 50.0, 200.0]
    with Session(migrated_runtime["engine"]) as session:
        for identifier, risk, value in zip(identifiers, risks, values, strict=True):
            customer = Customer(
                customer_id=identifier,
                country="STEP08 TEST COUNTRY",
                feature_as_of=now,
                feature_schema_version="vantara-churn-features-v1",
                feature_payload={},
                sequence_payload=None,
                net_spend=value,
                value_tier="high",
                source_sha256="0" * 64,
                loaded_at=now,
            )
            customer.segment = Segment(
                segment_id=98,
                segment_name="STEP08 Test Segment",
                model_version="step08-test",
                assigned_at=now,
            )
            customer.predictions.append(
                Prediction(
                    model_version="step08-test",
                    scored_at=now,
                    as_of_timestamp=now,
                    churn_probability=risk,
                    churn_label=risk >= 0.5,
                    churn_threshold=0.5,
                    churn_threshold_version="step08-test",
                    predicted_clv_180d=value,
                    next_purchase_probability=None,
                    next_category_id=None,
                    next_category_probability=None,
                    anomaly_score=0.01,
                    anomaly_flag=False,
                )
            )
            session.add(customer)
        session.commit()
    try:
        with _client(migrated_runtime) as client:
            response = client.get(
                "/api/v1/analytics/churn-priority",
                params={"country": "STEP08 TEST COUNTRY", "limit": 10},
            )
        assert response.status_code == 200
        rows = response.json()
        assert [row["customer_id"] for row in rows] == ["STEP08-C", "STEP08-B", "STEP08-A"]
        assert rows[0]["retention_priority"] == pytest.approx(0.5)
        assert rows[1]["retention_priority"] == pytest.approx(0.0)
        assert rows[2]["retention_priority"] == pytest.approx(0.0)
        assert rows[0]["churn_probability"] == pytest.approx(0.5)
        assert rows[0]["predicted_clv_180d"] == pytest.approx(200.0)
    finally:
        with Session(migrated_runtime["engine"]) as session:
            for identifier in identifiers:
                customer = session.get(Customer, identifier)
                if customer is not None:
                    session.delete(customer)
            session.commit()


def test_minmax_constant_population_is_deterministic() -> None:
    """A constant non-empty population remains fully normalized and comparable."""
    assert _minmax([]) == []
    assert _minmax([4.0, 4.0]) == [1.0, 1.0]
