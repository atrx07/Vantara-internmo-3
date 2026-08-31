"""STEP 05 deep-learning, grouped-snapshot, and artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.ann import ChurnANN, load_ann
from src.models.autoencoder import BehavioralAutoencoder, load_autoencoder
from src.models.purchase_lstm import (
    PurchaseLSTM,
    build_rolling_sequences,
    grouped_cv_indices,
    load_purchase_lstm,
)
from src.models.step05_pipeline import run_step05
from src.models.torch_common import fit_numeric_transform


def test_torch_model_shapes_and_training_only_transform() -> None:
    """Architectures emit one value per row and transform statistics are frozen."""
    transform = fit_numeric_transform(np.array([[1.0, np.nan], [3.0, 5.0]]))
    transformed = transform.transform(np.array([[100.0, np.nan]]))
    assert transformed.shape == (1, 2)
    assert transform.medians.tolist() == [2.0, 5.0]

    ann = ChurnANN(4, 0.3, 0.2)
    ann.eval()
    assert ann(torch.ones((2, 4))).shape == (2,)

    lstm = PurchaseLSTM(4, 2, 5, 0.2)
    lstm.eval()
    logits = lstm(
        torch.ones((2, 3, 2)),
        torch.tensor([[1, 2, 0], [2, 1, 3]]),
        torch.tensor([2, 3]),
    )
    assert logits.shape == (2,)

    autoencoder = BehavioralAutoencoder(4, 3, 2)
    assert autoencoder(torch.ones((2, 4))).shape == (2, 4)


def _write_step05_project(root: Path) -> Path:
    data_interim = root / "data" / "interim"
    data_processed = root / "data" / "processed"
    artifacts = root / "models_artifacts"
    data_interim.mkdir(parents=True)
    data_processed.mkdir(parents=True)
    artifacts.mkdir(parents=True)

    customers = [f"C{index:02d}" for index in range(12)]
    partitions = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2
    split = pd.DataFrame(
        {
            "customer_id": customers,
            "partition": partitions,
            "split_seed": 42,
            "split_version": "test-split-v1",
        }
    )
    split.to_parquet(data_processed / "customer_split.parquet", index=False)

    rng = np.random.default_rng(42)
    churn = pd.DataFrame(
        {
            "customer_id": customers,
            "feature_a": rng.normal(size=12),
            "feature_b": rng.normal(size=12),
            "feature_c": rng.normal(size=12),
            "feature_d": rng.normal(size=12),
            "churn": [0, 1] * 6,
            "partition": partitions,
            "split_version": "test-split-v1",
        }
    )
    churn.to_parquet(data_processed / "churn.parquet", index=False)
    taxonomy = pd.DataFrame(
        {
            "stock_code": ["P0", "P1"],
            "category_id": [0, 1],
            "taxonomy_version": ["test-taxonomy", "test-taxonomy"],
        }
    )
    taxonomy.to_parquet(data_processed / "taxonomy.parquet", index=False)

    transaction_rows: list[dict[str, object]] = []
    for customer_index, customer_id in enumerate(customers[:10]):
        months = range(1, 12) if customer_index % 2 == 0 else [1, 4, 7, 10]
        for month in months:
            transaction_rows.append(
                {
                    "customer_id": customer_id,
                    "invoice": f"{customer_id}-{month:02d}",
                    "stock_code": "P0" if month % 2 else "P1",
                    "invoice_date": pd.Timestamp(2021, month, 5),
                    "quantity": 1,
                    "gross_positive_value": float(10 + month),
                    "is_positive_purchase": True,
                    "is_valid_merchandise": True,
                }
            )
    pd.DataFrame(transaction_rows).to_parquet(data_interim / "transactions.parquet", index=False)
    metadata = {
        "source_sha256": "a" * 64,
        "observation_end": "2021-12-31 00:00:00",
    }
    (data_processed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    schema = {
        "schema_version": "test-features-v1",
        "feature_count": 4,
        "selected_features": ["feature_a", "feature_b", "feature_c", "feature_d"],
    }
    (artifacts / "schema.json").write_text(json.dumps(schema), encoding="utf-8")

    config = {
        "project": {"random_seed": 42},
        "logging": {"level": "INFO"},
        "cleaning": {"interim_transactions": "data/interim/transactions.parquet"},
        "outputs": {
            "customer_split": "data/processed/customer_split.parquet",
            "churn_features": "data/processed/churn.parquet",
            "product_taxonomy": "data/processed/taxonomy.parquet",
            "step02_metadata": "data/processed/metadata.json",
        },
        "analysis": {"churn_feature_schema": "models_artifacts/schema.json"},
        "deep_learning": {
            "experiment_name": "step05-test",
            "tracking_directory": "mlruns",
            "artifact_directory": "models_artifacts/step05",
            "evidence_directory": "reports/deep_learning",
            "torch_threads": 1,
            "ann": {
                "batch_size": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout_first": 0.1,
                "dropout_second": 0.1,
                "max_epochs": 2,
                "patience": 2,
                "minimum_delta": 0.0,
            },
            "lstm": {
                "horizon_days": 30,
                "sequence_length": 4,
                "snapshot_frequency": "MS",
                "max_snapshots_per_customer": 3,
                "minimum_history_events": 2,
                "embedding_size": 2,
                "hidden_size": 4,
                "dropout": 0.1,
                "batch_size": 8,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "cv_folds": 2,
                "cv_max_epochs": 1,
                "max_epochs": 2,
                "patience": 2,
                "minimum_delta": 0.0,
            },
            "autoencoder": {
                "features": ["feature_a", "feature_b"],
                "hidden_size": 4,
                "latent_size": 2,
                "batch_size": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "max_epochs": 2,
                "patience": 2,
                "minimum_delta": 0.0,
                "threshold_percentile": 99.0,
            },
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_step05_pipeline_and_group_isolation(tmp_path: Path) -> None:
    """Synthetic orchestration trains, logs, reloads, and never includes test customers."""
    config_path = _write_step05_project(tmp_path)
    summary = run_step05(config_path, project_root=tmp_path)
    assert summary["held_out_test_accessed"] is False
    assert summary["feature_count"] == 4
    assert summary["lstm"]["cv_folds"] == 2
    assert summary["mlflow_run_count"] == 5
    assert len(summary["artifact_reload_checks"]) == 3

    artifact_directory = tmp_path / "models_artifacts" / "step05"
    ann, ann_metadata = load_ann(artifact_directory / "churn_ann.pt")
    lstm, lstm_metadata = load_purchase_lstm(artifact_directory / "purchase_lstm.pt")
    autoencoder, autoencoder_metadata = load_autoencoder(
        artifact_directory / "behavioral_autoencoder.pt"
    )
    assert isinstance(ann, ChurnANN)
    assert isinstance(lstm, PurchaseLSTM)
    assert isinstance(autoencoder, BehavioralAutoencoder)
    assert ann_metadata["held_out_test_accessed"] is False
    assert lstm_metadata["held_out_test_accessed"] is False
    assert autoencoder_metadata["held_out_test_accessed"] is False

    split = pd.read_parquet(tmp_path / "data" / "processed" / "customer_split.parquet")
    transactions = pd.read_parquet(tmp_path / "data" / "interim" / "transactions.parquet")
    taxonomy = pd.read_parquet(tmp_path / "data" / "processed" / "taxonomy.parquet")
    sequences = build_rolling_sequences(
        transactions,
        taxonomy,
        split,
        observation_end=pd.Timestamp("2021-12-31"),
        config=yaml.safe_load(config_path.read_text(encoding="utf-8"))["deep_learning"]["lstm"],
    )
    assert not set(sequences.customer_ids).intersection({"C10", "C11"})
    train_mask = sequences.partitions == "train"
    folds = grouped_cv_indices(
        sequences.labels[train_mask],
        sequences.customer_ids[train_mask],
        n_splits=2,
        seed=42,
    )
    for train_indices, validation_indices in folds:
        assert set(sequences.customer_ids[train_mask][train_indices]).isdisjoint(
            sequences.customer_ids[train_mask][validation_indices]
        )
