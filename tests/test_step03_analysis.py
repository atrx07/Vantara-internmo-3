"""STEP 03 EDA, schema-freeze, evidence, and M1 orchestration tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.eda import build_eda_evidence, run_step03_analysis
from src.features.schema import freeze_churn_feature_schema
from src.pipeline import run_pipeline


def _churn_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 60
    net_spend = rng.lognormal(mean=6.0, sigma=0.8, size=rows)
    partitions = np.array(["train"] * 42 + ["validation"] * 9 + ["test"] * 9)
    return pd.DataFrame(
        {
            "customer_id": pd.Series([f"CUST{index:03d}" for index in range(rows)], dtype="string"),
            "cutoff_timestamp": pd.Timestamp("2021-05-01"),
            "frequency_orders": rng.integers(1, 20, size=rows),
            "gross_spend": net_spend * 1.01,
            "avg_order_value": rng.lognormal(mean=4.0, sigma=0.4, size=rows),
            "recency_days": rng.uniform(1, 300, size=rows),
            "net_spend": net_spend,
            "historical_customer_value": net_spend,
            "return_rate": rng.uniform(0, 0.3, size=rows),
            "category_affinity_unknown": rng.uniform(0, 0.1, size=rows),
            "category_affinity_00": rng.uniform(0, 1, size=rows),
            "category_affinity_01": rng.uniform(0, 1, size=rows),
            "engagement_score": rng.uniform(0, 100, size=rows),
            "churn": np.tile([0, 1], rows // 2).astype("int8"),
            "partition": pd.Categorical(partitions, categories=["train", "validation", "test"]),
            "split_version": pd.Series(["test-v1"] * rows, dtype="string"),
        }
    )


def test_schema_freeze_is_training_only_and_removes_redundancy() -> None:
    features = _churn_features()
    baseline = freeze_churn_feature_schema(
        features,
        schema_version="test-schema-v1",
        correlation_threshold=0.95,
        vif_threshold=10.0,
    )
    attacked = features.copy()
    attacked.loc[attacked["partition"] != "train", "recency_days"] = 1e12
    after_attack = freeze_churn_feature_schema(
        attacked,
        schema_version="test-schema-v1",
        correlation_threshold=0.95,
        vif_threshold=10.0,
    )

    assert baseline.selected_features == after_attack.selected_features
    pd.testing.assert_frame_equal(baseline.final_vif, after_attack.final_vif)
    assert "historical_customer_value" in baseline.exclusions
    assert "gross_spend" in baseline.exclusions
    assert "category_affinity_unknown" in baseline.exclusions
    assert baseline.final_vif["vif"].max() <= 10.0


def test_eda_contains_every_required_analysis_and_hypotheses(
    step02_transactions: pd.DataFrame,
) -> None:
    evidence = build_eda_evidence(
        step02_transactions,
        _churn_features(),
        top_countries=10,
    )

    assert set(evidence.rfm_summary["feature"]) >= {
        "recency_days",
        "frequency_orders",
        "net_spend",
    }
    assert evidence.class_balance["customers"].sum() == 60
    assert not evidence.country_summary.empty
    assert not evidence.seasonality_summary.empty
    assert "is_statistical_outlier" in set(evidence.outlier_summary["quality_flag"])
    assert len(evidence.hypotheses) == 7


def test_step03_analysis_writes_schema_tables_figures_and_summary(
    tmp_path: Path,
    step02_transactions: pd.DataFrame,
) -> None:
    interim = tmp_path / "data/interim"
    processed = tmp_path / "data/processed"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    step02_transactions.to_parquet(interim / "transactions_clean.parquet", index=False)
    _churn_features().to_parquet(processed / "customer_features_churn.parquet", index=False)
    config = {
        "cleaning": {"interim_transactions": "data/interim/transactions_clean.parquet"},
        "outputs": {"churn_features": "data/processed/customer_features_churn.parquet"},
        "analysis": {
            "evidence_directory": "reports/eda",
            "data_freeze_directory": "reports/data_freeze",
            "churn_feature_schema": "models_artifacts/churn_feature_schema.json",
            "correlation_threshold": 0.95,
            "vif_threshold": 10.0,
            "feature_schema_version": "test-schema-v1",
            "top_countries": 10,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = run_step03_analysis(config_path, project_root=tmp_path)

    assert summary["schema_version"] == "test-schema-v1"
    assert summary["maximum_final_vif"] <= 10.0
    assert (tmp_path / "models_artifacts/churn_feature_schema.json").is_file()
    assert (tmp_path / "reports/data_freeze/correlation_matrix.csv").is_file()
    assert (tmp_path / "reports/eda/rfm_distributions.png").is_file()
    assert (tmp_path / "reports/eda/hypotheses.md").is_file()


def test_m1_pipeline_runs_step02_then_step03(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_step02(config_path: str | Path) -> dict[str, object]:
        del config_path
        calls.append("step02")
        return {"rows": 10}

    def fake_step03(config_path: str | Path) -> dict[str, object]:
        del config_path
        calls.append("step03")
        return {"customer_rows": 5}

    monkeypatch.setattr("src.pipeline.run_step02", fake_step02)
    monkeypatch.setattr("src.pipeline.run_step03_analysis", fake_step03)

    result = run_pipeline("config.yaml")

    assert calls == ["step02", "step03"]
    assert result == {"step02": {"rows": 10}, "step03": {"customer_rows": 5}}
