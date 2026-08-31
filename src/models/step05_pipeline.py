"""Execute governed STEP 05 PyTorch training and evidence generation."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from src.data.validation import DataValidationError
from src.models.ann import load_ann, train_ann
from src.models.autoencoder import load_autoencoder, train_autoencoder
from src.models.common import configure_mlflow, load_feature_schema
from src.models.purchase_lstm import (
    RollingSequences,
    build_rolling_sequences,
    load_purchase_lstm,
    train_purchase_lstm,
)
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"Required STEP 05 input does not exist: {path}")
    return pd.read_parquet(path)


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype("string").dropna().unique()
    if len(values) != 1:
        raise DataValidationError(f"Expected exactly one {column}, got {list(values)}")
    return str(values[0])


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _export_runs(run_ids: Sequence[str], path: Path) -> pd.DataFrame:
    client = MlflowClient()
    rows: list[dict[str, Any]] = []
    for run_id in sorted(set(run_ids)):
        run = client.get_run(run_id)
        row: dict[str, Any] = {
            "run_id": run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "model_family": run.data.tags.get("model_family", ""),
            "model_name": run.data.tags.get("model_name", ""),
            "status": run.info.status,
        }
        row.update({f"metric_{key}": value for key, value in run.data.metrics.items()})
        row.update({f"parameter_{key}": value for key, value in run.data.params.items()})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["model_family", "run_name", "run_id"], ignore_index=True
    )
    frame.to_csv(path, index=False)
    return frame


def _scale_from_metadata(values: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    transform = metadata["transform"]
    medians = np.asarray(transform["medians"], dtype="float64")
    means = np.asarray(transform["means"], dtype="float64")
    scales = np.asarray(transform["scales"], dtype="float64")
    filled = np.where(np.isnan(values), medians, values)
    return ((filled - means) / scales).astype("float32")


def _reload_smoke(
    artifact_directory: Path,
    churn_table: pd.DataFrame,
    sequences: RollingSequences,
) -> list[dict[str, Any]]:
    validation = churn_table.loc[churn_table["partition"].astype("string").eq("validation")].head(1)
    checks: list[dict[str, Any]] = []

    ann, ann_metadata = load_ann(artifact_directory / "churn_ann.pt")
    ann_values = validation.loc[:, ann_metadata["feature_names"]].to_numpy(dtype="float64")
    ann_values = _scale_from_metadata(ann_values, ann_metadata)
    with torch.no_grad():
        ann_probability = float(torch.sigmoid(ann(torch.from_numpy(ann_values)))[0])
    checks.append({"artifact": "churn_ann.pt", "reload": "PASS", "sample_output": ann_probability})

    lstm, lstm_metadata = load_purchase_lstm(artifact_directory / "purchase_lstm.pt")
    validation_index = int(np.flatnonzero(sequences.partitions == "validation")[0])
    length = int(sequences.lengths[validation_index])
    continuous = sequences.continuous[[validation_index]].copy()
    means = np.asarray(lstm_metadata["continuous_means"], dtype="float32")
    scales = np.asarray(lstm_metadata["continuous_scales"], dtype="float32")
    continuous[0, :length] = (continuous[0, :length] - means) / scales
    with torch.no_grad():
        lstm_logit = lstm(
            torch.from_numpy(continuous),
            torch.from_numpy(sequences.categories[[validation_index]]),
            torch.tensor([length], dtype=torch.int64),
        )
        lstm_probability = float(torch.sigmoid(lstm_logit)[0])
    checks.append(
        {
            "artifact": "purchase_lstm.pt",
            "reload": "PASS",
            "sample_output": lstm_probability,
        }
    )

    autoencoder, autoencoder_metadata = load_autoencoder(
        artifact_directory / "behavioral_autoencoder.pt"
    )
    autoencoder_values = validation.loc[:, autoencoder_metadata["feature_names"]].to_numpy(
        dtype="float64"
    )
    autoencoder_values = _scale_from_metadata(autoencoder_values, autoencoder_metadata)
    with torch.no_grad():
        values_tensor = torch.from_numpy(autoencoder_values)
        reconstruction_error = float(
            torch.mean((autoencoder(values_tensor) - values_tensor) ** 2, dim=1)[0]
        )
    checks.append(
        {
            "artifact": "behavioral_autoencoder.pt",
            "reload": "PASS",
            "sample_output": reconstruction_error,
        }
    )
    return checks


def run_step05(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run STEP 05 against train/validation data without evaluating final test."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    seed = int(config["project"]["random_seed"])
    deep_learning = config["deep_learning"]
    torch.set_num_threads(int(deep_learning["torch_threads"]))
    artifact_directory = resolve_project_path(
        deep_learning["artifact_directory"], project_root=root
    )
    evidence_directory = resolve_project_path(
        deep_learning["evidence_directory"], project_root=root
    )
    tracking_directory = resolve_project_path(
        deep_learning["tracking_directory"], project_root=root
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    configure_mlflow(tracking_directory, str(deep_learning["experiment_name"]))

    output_paths = {
        key: resolve_project_path(value, project_root=root)
        for key, value in config["outputs"].items()
    }
    churn_table = _load_frame(output_paths["churn_features"])
    split = _load_frame(output_paths["customer_split"])
    taxonomy = _load_frame(output_paths["product_taxonomy"])
    transactions = _load_frame(
        resolve_project_path(config["cleaning"]["interim_transactions"], project_root=root)
    )
    source_metadata = json.loads(output_paths["step02_metadata"].read_text(encoding="utf-8"))
    source_sha256 = str(source_metadata["source_sha256"])
    observation_end = pd.Timestamp(source_metadata["observation_end"])
    split_version = _single_value(churn_table, "split_version")
    schema = load_feature_schema(
        resolve_project_path(config["analysis"]["churn_feature_schema"], project_root=root)
    )
    feature_names = tuple(str(value) for value in schema["selected_features"])
    schema_version = str(schema["schema_version"])
    base_metadata = {
        "source_sha256": source_sha256,
        "feature_schema_version": schema_version,
        "split_version": split_version,
        "seed": seed,
        "torch_version": str(torch.__version__),
    }
    tags = {
        "source_sha256": source_sha256,
        "feature_schema_version": schema_version,
        "split_version": split_version,
        "seed": str(seed),
        "held_out_test_accessed": "false",
    }

    ann_result = train_ann(
        churn_table,
        feature_names=feature_names,
        config=deep_learning["ann"],
        metadata=base_metadata,
        artifact_path=artifact_directory / "churn_ann.pt",
        mlflow_tags=tags,
    )
    sequences = build_rolling_sequences(
        transactions,
        taxonomy,
        split,
        observation_end=observation_end,
        config=deep_learning["lstm"],
    )
    category_count = int(taxonomy["category_id"].max()) + 2
    lstm_result = train_purchase_lstm(
        sequences,
        category_count=category_count,
        config=deep_learning["lstm"],
        metadata={**base_metadata, "sequence_schema_version": "vantara-purchase-sequence-v1"},
        artifact_path=artifact_directory / "purchase_lstm.pt",
        mlflow_tags=tags,
    )
    autoencoder_features = tuple(str(value) for value in deep_learning["autoencoder"]["features"])
    autoencoder_result = train_autoencoder(
        churn_table,
        feature_names=autoencoder_features,
        config=deep_learning["autoencoder"],
        metadata={**base_metadata, "feature_schema_version": "vantara-autoencoder-features-v1"},
        artifact_path=artifact_directory / "behavioral_autoencoder.pt",
        mlflow_tags=tags,
    )

    pd.DataFrame(ann_result["history"]).to_csv(
        evidence_directory / "ann_loss_history.csv", index=False
    )
    _write_json(evidence_directory / "ann_validation_metrics.json", ann_result["metrics"])
    pd.DataFrame(lstm_result["history"]).to_csv(
        evidence_directory / "lstm_loss_history.csv", index=False
    )
    pd.DataFrame(lstm_result["cv"]).to_csv(evidence_directory / "lstm_grouped_cv.csv", index=False)
    _write_json(evidence_directory / "lstm_validation_metrics.json", lstm_result["metrics"])
    pd.DataFrame(autoencoder_result["history"]).to_csv(
        evidence_directory / "autoencoder_loss_history.csv", index=False
    )
    pd.DataFrame(autoencoder_result["feature_contributions"]).to_csv(
        evidence_directory / "autoencoder_feature_contributions.csv", index=False
    )
    pd.DataFrame(autoencoder_result["validation_errors"]).to_csv(
        evidence_directory / "autoencoder_validation_errors.csv", index=False
    )
    _write_json(
        evidence_directory / "autoencoder_reconstruction_summary.json",
        autoencoder_result["metrics"],
    )

    run_ids = [
        str(ann_result["metadata"]["mlflow_run_id"]),
        str(lstm_result["metadata"]["mlflow_run_id"]),
        str(autoencoder_result["metadata"]["mlflow_run_id"]),
        *(str(row["mlflow_run_id"]) for row in lstm_result["cv"]),
    ]
    mlflow_runs = _export_runs(run_ids, evidence_directory / "mlflow_run_summary.csv")
    reload_checks = _reload_smoke(artifact_directory, churn_table, sequences)
    _write_json(evidence_directory / "artifact_reload_smoke.json", reload_checks)
    train_sequence_mask = sequences.partitions == "train"
    validation_sequence_mask = sequences.partitions == "validation"
    cv_frame = pd.DataFrame(lstm_result["cv"])
    summary: dict[str, Any] = {
        "step": "STEP 05",
        **base_metadata,
        "feature_count": len(feature_names),
        "held_out_test_accessed": False,
        "ann": {
            "metrics": ann_result["metrics"],
            "artifact": ann_result["artifact"],
            "epochs_ran": ann_result["metadata"]["epochs_ran"],
            "best_epoch": ann_result["metadata"]["best_epoch"],
        },
        "lstm": {
            "metrics": lstm_result["metrics"],
            "artifact": lstm_result["artifact"],
            "cv_folds": len(lstm_result["cv"]),
            "cv_mean_roc_auc": float(cv_frame["roc_auc"].mean()),
            "cv_mean_recall": float(cv_frame["recall"].mean()),
            "cv_mean_f1": float(cv_frame["f1"].mean()),
            "train_snapshots": int(train_sequence_mask.sum()),
            "validation_snapshots": int(validation_sequence_mask.sum()),
            "train_customers": len(set(sequences.customer_ids[train_sequence_mask])),
            "validation_customers": len(set(sequences.customer_ids[validation_sequence_mask])),
            "sequence_length": int(deep_learning["lstm"]["sequence_length"]),
            "horizon_days": int(deep_learning["lstm"]["horizon_days"]),
        },
        "autoencoder": {
            "metrics": autoencoder_result["metrics"],
            "artifact": autoencoder_result["artifact"],
            "feature_count": len(autoencoder_features),
            "interpretation": "manual-review anomaly candidate; not confirmed fraud",
        },
        "mlflow_run_count": len(mlflow_runs),
        "artifact_reload_checks": reload_checks,
    }
    _write_json(evidence_directory / "step05_summary.json", summary)
    LOGGER.info(
        "STEP 05 deep-learning pipeline passed",
        extra={"event": "step05_pipeline_passed", "mlflow_runs": len(mlflow_runs)},
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the STEP 05 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the STEP 05 command-line pipeline."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config["logging"]["level"]))
    run_step05(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
