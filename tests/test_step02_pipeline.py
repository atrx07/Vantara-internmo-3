"""Small end-to-end STEP 02 orchestration and persistence test."""

from pathlib import Path

import pandas as pd
import yaml

from src.data.contracts import IngestionSummary
from src.data.step02_pipeline import run_step02


def test_step02_pipeline_persists_deterministic_outputs(
    tmp_path: Path,
    monkeypatch: object,
    step02_transactions: pd.DataFrame,
    cleaning_config: dict[str, object],
) -> None:
    config = {
        "project": {"random_seed": 42},
        "logging": {"level": "INFO"},
        "cleaning": {
            **cleaning_config,
            "interim_transactions": "data/interim/transactions_clean.parquet",
            "outlier_audit": "data/interim/outlier_audit.parquet",
        },
        "snapshots": {
            "churn_horizon_days": 15,
            "clv_horizon_days": 30,
            "trend_window_days": 30,
        },
        "features": {
            "markdown_price_ratio": 0.90,
            "markdown_min_observations": 1,
            "taxonomy": {
                "version": "test-taxonomy-v1",
                "candidate_clusters": [2, 3],
                "max_tfidf_features": 100,
                "svd_components": 5,
                "silhouette_sample_size": 100,
                "min_cluster_share": 0.01,
            },
            "split": {
                "version": "test-split-v1",
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "seed": 42,
            },
        },
        "outputs": {
            "customer_split": "data/processed/customer_split.parquet",
            "churn_features": "data/processed/customer_features_churn.parquet",
            "clv_features": "data/processed/customer_features_clv.parquet",
            "product_taxonomy": "data/processed/product_taxonomy.parquet",
            "product_reference_prices": "data/processed/product_reference_prices.parquet",
            "product_frequency_encoding": "data/processed/product_frequency_encoding.parquet",
            "preprocessing_artifacts": "data/processed/preprocessing_contracts.joblib",
            "step02_metadata": "data/processed/step02_metadata.json",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = IngestionSummary(
        path="test.xlsx",
        sha256="a" * 64,
        sheet_rows={"first": len(step02_transactions)},
        combined_rows=len(step02_transactions),
        columns=tuple(step02_transactions.columns[:8]),
        minimum_date=str(step02_transactions["invoice_date"].min()),
        maximum_date=str(step02_transactions["invoice_date"].max()),
        chronological=True,
    )

    def fake_load_transactions(
        config_path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> tuple[pd.DataFrame, IngestionSummary]:
        del config_path, project_root
        return step02_transactions.copy(), summary

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "src.data.step02_pipeline.load_transactions", fake_load_transactions
    )
    first = run_step02(config_path, project_root=tmp_path)
    second = run_step02(config_path, project_root=tmp_path)

    assert first["fingerprints"] == second["fingerprints"]
    assert first["customer_count"] == 8
    assert sum(first["split_counts"].values()) == 8
    assert (tmp_path / "data/interim/transactions_clean.parquet").is_file()
    assert (tmp_path / "data/processed/customer_features_churn.parquet").is_file()
    assert (tmp_path / "data/processed/customer_features_clv.parquet").is_file()
    assert (tmp_path / "data/processed/preprocessing_contracts.joblib").is_file()
