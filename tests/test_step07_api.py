"""STEP 07 FastAPI contract and real scoring/persistence integration tests."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from api.main import create_app
from api.models import Customer, Prediction, Transaction


def _known_customer(runtime: dict[str, object]) -> str:
    with Session(runtime["engine"]) as session:
        customer = session.scalar(
            select(Customer)
            .where(Customer.sequence_payload.is_not(None))
            .order_by(Customer.customer_id)
            .limit(1)
        )
        assert customer is not None
        return customer.customer_id


def _client(runtime: dict[str, object]) -> TestClient:
    app = create_app(
        database_url=str(runtime["database_url"]),
        project_root=runtime["root"],
        artifact_root=runtime["root"] / "models_artifacts",
    )
    return TestClient(app)


def test_health_and_model_metadata(migrated_runtime: dict[str, object]) -> None:
    """Health and safe model metadata succeed with real DB/artifacts."""
    with _client(migrated_runtime) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json() == {
            "service": "vantara-api",
            "status": "ok",
            "database": "ready",
            "artifacts": "ready",
        }
        metadata = client.get("/api/v1/models/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["churn"]["model"] == "random_forest"
        assert metadata.json()["feature_count"] == 47


def test_known_customer_score_persists_without_per_request_reload(
    migrated_runtime: dict[str, object],
) -> None:
    """Real frozen artifacts score once per request while artifact files remain startup-only."""
    customer_id = _known_customer(migrated_runtime)
    with Session(migrated_runtime["engine"]) as session:
        before = int(
            session.scalar(
                select(func.count())
                .select_from(Prediction)
                .where(Prediction.customer_id == customer_id)
            )
            or 0
        )
    with _client(migrated_runtime) as client:
        with patch("api.artifacts.joblib.load", side_effect=AssertionError("unexpected reload")):
            response = client.post("/api/v1/predict/customer", json={"customer_id": customer_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["persisted"] is True
    assert payload["prediction_id"] is not None
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["predicted_clv_180d"] >= 0
    assert 0 <= payload["next_purchase_probability"] <= 1
    assert payload["segment_name"]
    with Session(migrated_runtime["engine"]) as session:
        after = int(
            session.scalar(
                select(func.count())
                .select_from(Prediction)
                .where(Prediction.customer_id == customer_id)
            )
            or 0
        )
    assert after == before + 1


def test_unknown_malformed_and_unsupported_as_of_paths(
    migrated_runtime: dict[str, object],
) -> None:
    """Unknown identifiers and invalid Pydantic/as-of inputs return clear 4xx errors."""
    customer_id = _known_customer(migrated_runtime)
    with _client(migrated_runtime) as client:
        assert (
            client.post("/api/v1/predict/customer", json={"customer_id": "UNKNOWN"}).status_code
            == 404
        )
        assert client.post("/api/v1/predict/customer", json={"customer_id": ""}).status_code == 422
        unsupported = client.post(
            "/api/v1/predict/customer",
            json={"customer_id": customer_id, "as_of_date": "2020-01-01T00:00:00"},
        )
        assert unsupported.status_code == 422
        assert "persisted as-of timestamp" in unsupported.json()["detail"]


def test_customer_explanation_recommendation_and_segment_paths(
    migrated_runtime: dict[str, object],
) -> None:
    """Dashboard-supporting customer XAI, recommendation, and segment reads work."""
    customer_id = _known_customer(migrated_runtime)
    with _client(migrated_runtime) as client:
        customer = client.get(f"/api/v1/customers/{customer_id}")
        explanation = client.get(f"/api/v1/customers/{customer_id}/explanation")
        recommendations = client.get(f"/api/v1/customers/{customer_id}/recommendations")
        segments = client.get("/api/v1/segments")
    assert customer.status_code == 200
    assert explanation.status_code == 200
    assert explanation.json()["method"] == "TreeSHAP"
    assert explanation.json()["causal"] is False
    assert len(recommendations.json()) == 5
    assert recommendations.json()[0]["rank"] == 1
    assert len(segments.json()) == 8
    segment_id = customer.json()["segment_id"]
    with _client(migrated_runtime) as client:
        segment_customers = client.get(
            f"/api/v1/segments/{segment_id}", params={"value_tier": customer.json()["value_tier"]}
        )
    assert segment_customers.status_code == 200
    assert any(row["customer_id"] == customer_id for row in segment_customers.json())


def test_revenue_analytics_uses_persisted_transactions(
    migrated_runtime: dict[str, object],
) -> None:
    """Revenue analytics aggregates valid persisted merchandise by calendar month."""
    customer_id = _known_customer(migrated_runtime)
    with Session(migrated_runtime["engine"]) as session:
        session.add(
            Transaction(
                invoice="API-REVENUE-1",
                stock_code="85123A",
                customer_id=customer_id,
                description="VALID MERCHANDISE",
                quantity=2,
                price=3.5,
                invoice_timestamp=datetime(2011, 1, 5, 10),
                country="United Kingdom",
                is_product=True,
                is_return=False,
                is_valid_merchandise=True,
            )
        )
        session.commit()
    try:
        with _client(migrated_runtime) as client:
            response = client.get("/api/v1/analytics/revenue")
        assert response.status_code == 200
        assert {row["period"]: row["revenue"] for row in response.json()}["2011-01"] == 7.0
    finally:
        with Session(migrated_runtime["engine"]) as session:
            session.execute(delete(Transaction).where(Transaction.invoice == "API-REVENUE-1"))
            session.commit()


def test_valid_and_malformed_batch_upload(migrated_runtime: dict[str, object]) -> None:
    """Canonical transaction history is scored/persisted while malformed schema fails."""
    canonical = (
        "invoice,stock_code,description,quantity,invoice_date,price,customer_id,country\n"
        "B001,85123A,WHITE HANGING HEART T-LIGHT HOLDER,2,"
        "2020-01-01T10:00:00,2.55,NEW-001,United Kingdom\n"
        "B002,85123A,WHITE HANGING HEART T-LIGHT HOLDER,3,"
        "2020-02-01T10:00:00,2.55,NEW-001,United Kingdom\n"
    )
    try:
        with _client(migrated_runtime) as client:
            with (
                patch("api.artifacts.joblib.load", side_effect=AssertionError("unexpected reload")),
                patch(
                    "api.artifacts.pd.read_parquet",
                    side_effect=AssertionError("unexpected reload"),
                ),
            ):
                response = client.post(
                    "/api/v1/predict/batch",
                    files={"file": ("batch.csv", canonical, "text/csv")},
                )
                malformed = client.post(
                    "/api/v1/predict/batch",
                    files={"file": ("bad.csv", "customer_id\nNEW-002\n", "text/csv")},
                )
        assert response.status_code == 200
        assert response.json()[0]["customer_id"] == "NEW-001"
        assert response.json()[0]["persisted"] is True
        assert malformed.status_code == 422
        assert "columns must exactly match" in malformed.json()["detail"]
        with Session(migrated_runtime["engine"]) as session:
            assert session.get(Customer, "NEW-001") is not None
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(Prediction)
                    .where(Prediction.customer_id == "NEW-001")
                )
                == 1
            )
    finally:
        with Session(migrated_runtime["engine"]) as session:
            customer = session.get(Customer, "NEW-001")
            if customer is not None:
                session.delete(customer)
                session.commit()
