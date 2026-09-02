"""STEP 07 migration, schema, artifact, and initialization tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from api.artifacts import ArtifactCompatibilityError, ArtifactRegistry
from api.initialization import initialize_serving_data
from api.models import Customer, Prediction, Recommendation, Segment, Transaction


def test_fresh_migration_creates_required_schema(migrated_runtime: dict[str, object]) -> None:
    """A new database receives all governed core and approved-extension tables."""
    engine = migrated_runtime["engine"]
    assert set(inspect(engine).get_table_names()) == {
        "alembic_version",
        "customers",
        "transactions",
        "predictions",
        "segments",
        "recommendations",
    }
    assert {index["name"] for index in inspect(engine).get_indexes("predictions")} == {
        "ix_predictions_customer_scored",
        "ix_predictions_risk_value",
    }


def test_schema_compiles_for_postgresql() -> None:
    """All ORM tables compile with the locked PostgreSQL dialect."""
    for table in (Customer, Transaction, Prediction, Segment, Recommendation):
        statement = str(CreateTable(table.__table__).compile(dialect=postgresql.dialect()))
        assert "CREATE TABLE" in statement
        assert table.__tablename__ in statement


def test_real_artifact_initialization_is_complete_and_idempotent(
    migrated_runtime: dict[str, object],
) -> None:
    """Full governed customer state loads once without model retraining or duplication."""
    counts = migrated_runtime["counts"]
    assert counts == {
        "customers": 4952,
        "segments": 4952,
        "recommendations": 24760,
        "transactions": 0,
    }
    repeated = initialize_serving_data(
        migrated_runtime["engine"],
        migrated_runtime["root"],
        migrated_runtime["registry"],
        migrated_runtime["config"],
    )
    assert repeated == counts


def test_database_foreign_keys_and_customer_state(migrated_runtime: dict[str, object]) -> None:
    """Seeded segment/recommendation rows remain attached to valid customer keys."""
    with Session(migrated_runtime["engine"]) as session:
        assert session.scalar(select(func.count()).select_from(Customer)) == 4952
        assert session.scalar(select(func.count()).select_from(Segment)) == 4952
        assert session.scalar(select(func.count()).select_from(Recommendation)) == 24760
        customer = session.scalar(select(Customer).order_by(Customer.customer_id).limit(1))
        assert customer is not None
        assert customer.feature_schema_version == "vantara-churn-features-v1"
        assert len(customer.feature_payload) == 47
        assert len(customer.recommendations) == 5


def test_artifact_registry_fails_loudly_when_artifacts_are_missing(tmp_path: Path) -> None:
    """Startup must not continue with a missing or substituted artifact directory."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ArtifactCompatibilityError, match="Missing serving artifact"):
        ArtifactRegistry(root, tmp_path)
