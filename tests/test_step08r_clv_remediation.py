"""STEP 08R CLV v2 leakage, grouping, hurdle, artifact, and evidence tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression, Ridge
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.main import create_app
from api.models import Customer
from src.data.validation import DataValidationError
from src.models.clv_remediation import (
    CLV_V2_FEATURES,
    MODEL_VERSION,
    HurdleCLVRegressor,
    build_rolling_clv_dataset,
    candidate_estimators,
    grouped_clv_folds,
    verify_original_final_evidence,
    verify_original_v1_history,
)
from src.models.final_evaluate import _validate_final_lock
from src.utils.hashing import sha256_file


def _split() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.Series(["TRAIN-A", "TRAIN-B", "VALID-A", "TEST-A"], dtype="string"),
            "partition": pd.Categorical(["train", "train", "validation", "test"]),
            "split_seed": [42] * 4,
            "split_version": pd.Series(["vantara-customer-split-v1"] * 4, dtype="string"),
        }
    )


def _transactions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for customer_index, customer_id in enumerate(["TRAIN-A", "TRAIN-B", "VALID-A", "TEST-A"]):
        for month_index, timestamp in enumerate(
            pd.date_range("2020-01-01", "2021-12-01", freq="MS")
        ):
            value = float(10 + customer_index * 3 + month_index)
            rows.append(
                {
                    "customer_id": customer_id,
                    "invoice": f"{customer_id}-{month_index}",
                    "invoice_date": timestamp + pd.Timedelta(days=customer_index),
                    "quantity": 2 + month_index % 4,
                    "is_valid_merchandise": True,
                    "is_positive_purchase": True,
                    "signed_merchandise_value": value,
                    "gross_positive_value": value,
                }
            )
    return pd.DataFrame(rows)


def _remediation_config() -> dict[str, object]:
    return {
        "cv_folds": 5,
        "stratification_bins": 5,
        "catastrophic_fold_r2": -0.5,
        "tweedie_variance_power": 1.5,
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    }


def test_rolling_dataset_excludes_test_and_enforces_cutoff_and_horizon() -> None:
    """No original test customer, future feature row, or incomplete target may enter v2."""
    dataset, summary = build_rolling_clv_dataset(
        _transactions(),
        _split(),
        horizon_days=180,
        cutoff_frequency="MS",
        initial_history_days=120,
        minimum_customer_history_days=30,
        minimum_history_orders=2,
        bulk_order_units=10,
        high_value_order_gbp=100.0,
    )
    assert "TEST-A" not in set(dataset["customer_id"])
    assert set(dataset["partition"]) == {"train", "validation"}
    assert dataset["feature_max_timestamp"].lt(dataset["cutoff_timestamp"]).all()
    assert (
        dataset["target_window_end"] - dataset["cutoff_timestamp"] == pd.Timedelta(days=180)
    ).all()
    assert dataset["target_window_end"].le(dataset["observation_end"]).all()
    assert summary["original_test_customer_intersection"] == 0
    assert summary["all_features_strictly_before_cutoff"] is True
    assert summary["all_targets_have_complete_horizon"] is True


def test_grouped_folds_keep_every_customer_on_one_side() -> None:
    """All rolling snapshots for a customer remain isolated in five-fold validation."""
    records: list[dict[str, object]] = []
    for customer_index in range(25):
        for snapshot in range(3):
            record = {
                "customer_id": f"C{customer_index:02d}",
                "clv_180d_target": float((customer_index % 7) * 25 + snapshot),
                **{name: float(customer_index + snapshot) for name in CLV_V2_FEATURES},
            }
            records.append(record)
    table = pd.DataFrame(records)
    folds = grouped_clv_folds(table, n_splits=5, stratification_bins=5, seed=42)
    assert len(folds) == 5
    for train_index, validation_index in folds:
        train_customers = set(table.loc[train_index, "customer_id"])
        validation_customers = set(table.loc[validation_index, "customer_id"])
        assert train_customers.isdisjoint(validation_customers)


def test_candidate_imputation_is_fitted_on_training_rows_only() -> None:
    """The fold-local imputer cannot learn the held-out row's extreme value."""
    estimators = candidate_estimators(_remediation_config(), seed=42)
    estimator = estimators["ridge_log1p_baseline"]
    features = pd.DataFrame(np.ones((3, len(CLV_V2_FEATURES))), columns=CLV_V2_FEATURES)
    features.loc[1, CLV_V2_FEATURES[0]] = np.nan
    features.loc[2, CLV_V2_FEATURES[0]] = 1_000_000.0
    estimator.fit(features.iloc[:2], pd.Series([10.0, 20.0]))
    pipeline = estimator.estimator_.regressor_
    assert pipeline.named_steps["imputer"].statistics_[0] == pytest.approx(1.0)


def test_hurdle_handles_zero_and_positive_targets() -> None:
    """Hurdle occurrence and magnitude paths remain nonnegative, including all-zero data."""
    features = np.arange(12, dtype="float64").reshape(6, 2)
    hurdle = HurdleCLVRegressor(
        classifier=LogisticRegression(random_state=42), magnitude_model=Ridge(alpha=1.0)
    ).fit(features, np.asarray([0.0, 0.0, 10.0, 20.0, 30.0, 40.0]))
    probabilities = hurdle.positive_probability(features)
    predictions = hurdle.predict(features)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert (predictions >= 0).all()
    all_zero = HurdleCLVRegressor(
        classifier=LogisticRegression(random_state=42), magnitude_model=Ridge(alpha=1.0)
    ).fit(features, np.zeros(6))
    assert np.array_equal(all_zero.predict(features), np.zeros(6))


def test_v2_artifact_round_trip_is_compatible(tmp_path: Path) -> None:
    """A versioned CLV v2 bundle survives joblib reload and inference."""
    estimator = candidate_estimators(_remediation_config(), seed=42)["ridge_log1p_baseline"]
    features = pd.DataFrame(
        np.arange(12 * len(CLV_V2_FEATURES), dtype="float64").reshape(12, len(CLV_V2_FEATURES)),
        columns=CLV_V2_FEATURES,
    )
    estimator.fit(features, pd.Series(np.linspace(0, 110, 12)))
    path = tmp_path / "production_clv_v2.joblib"
    joblib.dump(
        {
            "pipeline": estimator,
            "metadata": {
                "model_version": MODEL_VERSION,
                "feature_names": list(CLV_V2_FEATURES),
                "original_held_out_test_reused": False,
            },
        },
        path,
    )
    reloaded = joblib.load(path)
    assert reloaded["metadata"]["model_version"] == MODEL_VERSION
    assert np.isfinite(reloaded["pipeline"].predict(features.head(2))).all()


def test_original_step06_evidence_is_unchanged_and_evaluator_stays_locked() -> None:
    """Historical test evidence matches the pre-remediation bytes and blocks reruns."""
    root = Path(__file__).resolve().parents[1]
    hashes = verify_original_final_evidence(root)
    assert len(hashes) == 7
    v1_hashes = verify_original_v1_history(root)
    assert len(v1_hashes) == 2
    with pytest.raises(DataValidationError, match="already started or completed"):
        _validate_final_lock(
            root,
            root / "reports" / "model_freeze" / "model_freeze.json",
            root / "reports" / "final_evaluation",
        )


def test_api_serving_registry_and_customer_payload_use_frozen_v2(
    migrated_runtime: dict[str, object],
) -> None:
    """Startup validates and serves CLV v2 while retaining the separate v1 artifact."""
    registry = migrated_runtime["registry"]
    assert registry.clv["metadata"]["model_version"] == MODEL_VERSION
    assert registry.clv["metadata"]["original_held_out_test_reused"] is False
    assert registry.safe_metadata()["clv"]["version"] == MODEL_VERSION
    assert Path("models_artifacts/clv/production_clv.joblib").is_file()
    with Session(migrated_runtime["engine"]) as session:
        customer = session.scalar(select(Customer).order_by(Customer.customer_id).limit(1))
        assert customer is not None
        assert set(CLV_V2_FEATURES).issubset(customer.feature_payload)
        score = registry.score(customer.feature_payload, customer.sequence_payload)
    assert score.predicted_clv_180d >= 0
    app = create_app(
        database_url=str(migrated_runtime["database_url"]),
        project_root=migrated_runtime["root"],
        artifact_root=Path(migrated_runtime["root"]) / "models_artifacts",
    )
    with TestClient(app) as client:
        metadata = client.get("/api/v1/models/metadata")
        response = client.post(
            "/api/v1/predict/customer", json={"customer_id": customer.customer_id}
        )
    assert metadata.status_code == 200
    assert metadata.json()["clv"]["version"] == MODEL_VERSION
    assert response.status_code == 200
    assert MODEL_VERSION in response.json()["model_version"]


def test_persisted_v2_dataset_and_freeze_evidence_match() -> None:
    """The real remediation dataset excludes test customers and matches the frozen artifact."""
    root = Path(__file__).resolve().parents[1]
    dataset = pd.read_parquet(root / "data" / "processed" / "clv_v2_rolling_snapshots.parquet")
    split = pd.read_parquet(root / "data" / "processed" / "customer_split.parquet")
    test_ids = set(
        split.loc[split["partition"].astype("string").eq("test"), "customer_id"].astype(str)
    )
    assert set(dataset["customer_id"].astype(str)).isdisjoint(test_ids)
    assert dataset["feature_max_timestamp"].lt(dataset["cutoff_timestamp"]).all()
    assert dataset["target_window_end"].le(dataset["observation_end"]).all()
    freeze = json.loads(
        (root / "reports" / "clv_remediation" / "production_clv_v2_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = root / str(freeze["artifact"])
    assert freeze["model_version"] == MODEL_VERSION
    assert freeze["original_held_out_test_reused"] is False
    assert sha256_file(artifact) == freeze["artifact_sha256"]
    for field, relative in {
        "candidate_comparison_sha256": "reports/clv_remediation/candidate_comparison.csv",
        "fold_metrics_sha256": "reports/clv_remediation/fold_metrics.csv",
        "dataset_summary_sha256": "reports/clv_remediation/dataset_summary.json",
    }.items():
        canonical_bytes = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical_bytes).hexdigest() == freeze[field]
