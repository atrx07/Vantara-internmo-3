"""Execute the single governed held-out test evaluation after model freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from src.data.validation import DataValidationError
from src.models.autoencoder import load_autoencoder
from src.models.next_category import build_next_category_targets
from src.models.purchase_lstm import build_rolling_sequences, load_purchase_lstm
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)
FREEZE_SENTINEL = "FINAL_TEST_AUTHORIZED_AND_CHOICES_FROZEN"


def classification_metrics_at_threshold(
    truth: Sequence[int], probabilities: Sequence[float], *, threshold: float
) -> dict[str, Any]:
    """Calculate governed binary metrics at an already-frozen threshold."""
    labels = np.asarray(truth, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    predictions = scores >= threshold
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1])
        .astype("int64")
        .tolist(),
        "threshold": float(threshold),
        "rows": len(labels),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _validate_final_lock(root: Path, freeze_path: Path, output: Path) -> dict[str, Any]:
    if (output / "final_metrics.json").exists() or (
        output / "final_evaluation_execution_lock.json"
    ).exists():
        raise DataValidationError(
            "Final held-out evaluation has already started or completed; rerun is forbidden"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("owner_step06_approval_received") or not freeze.get("choices_frozen"):
        raise DataValidationError("Final evaluation requires owner approval and frozen choices")
    if freeze.get("held_out_test_accessed") or freeze.get("final_test_status") != "NOT_RUN":
        raise DataValidationError("Freeze record does not show an untouched final test")
    status_text = (root / "STATUS.md").read_text(encoding="utf-8")
    if FREEZE_SENTINEL not in status_text:
        raise DataValidationError(
            f"STATUS.md must contain {FREEZE_SENTINEL} before final evaluation"
        )
    return freeze


def _score_lstm(
    root: Path,
    config: dict[str, Any],
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    split: pd.DataFrame,
    observation_end: pd.Timestamp,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sequences = build_rolling_sequences(
        transactions,
        taxonomy,
        split,
        observation_end=observation_end,
        config=config["deep_learning"]["lstm"],
        allowed_partitions=("test",),
    )
    if not np.all(sequences.partitions == "test"):
        raise DataValidationError("Non-test sequence entered final LSTM evaluation")
    model, metadata = load_purchase_lstm(
        root / "models_artifacts" / "next_purchase" / "purchase_lstm.pt"
    )
    means = np.asarray(metadata["continuous_means"], dtype="float32")
    scales = np.asarray(metadata["continuous_scales"], dtype="float32")
    continuous = sequences.continuous.copy()
    for index, length in enumerate(sequences.lengths):
        continuous[index, : int(length)] = (continuous[index, : int(length)] - means) / scales
    probabilities: list[np.ndarray] = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(continuous), batch_size):
            stop = start + batch_size
            logits = model(
                torch.from_numpy(continuous[start:stop]),
                torch.from_numpy(sequences.categories[start:stop]),
                torch.from_numpy(sequences.lengths[start:stop]),
            )
            probabilities.append(torch.sigmoid(logits).numpy())
    scores = np.concatenate(probabilities)
    metrics = classification_metrics_at_threshold(sequences.labels, scores, threshold=threshold)
    metrics["customers"] = len(set(sequences.customer_ids))
    predictions = pd.DataFrame(
        {
            "customer_id": sequences.customer_ids,
            "cutoff": sequences.cutoffs.astype(str),
            "truth": sequences.labels,
            "probability": scores,
            "prediction": (scores >= threshold).astype("int8"),
        }
    )
    return metrics, predictions


def _score_next_category(
    feature_table: pd.DataFrame,
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    test_features = feature_table.loc[feature_table["partition"].astype("string").eq("test")].copy()
    test_ids = set(test_features["customer_id"].astype("string"))
    test_transactions = transactions.loc[transactions["customer_id"].isin(test_ids)]
    cutoffs = test_features["cutoff_timestamp"].drop_duplicates()
    if len(cutoffs) != 1:
        raise DataValidationError("Final next-category features must share one cutoff")
    targets = build_next_category_targets(
        test_transactions, taxonomy, cutoff=pd.Timestamp(cutoffs.iloc[0])
    )
    modeling = test_features.merge(targets, on="customer_id", how="inner", validate="one_to_one")
    encoder = bundle["label_encoder"]
    unseen = sorted(set(modeling["next_category_id"]).difference(encoder.classes_))
    if unseen:
        raise DataValidationError(f"Final next-category data contain unseen labels: {unseen}")
    feature_names = list(bundle["metadata"]["feature_names"])
    truth = encoder.transform(modeling["next_category_id"].astype("int64"))
    probabilities = bundle["pipeline"].predict_proba(modeling.loc[:, feature_names])
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(encoder.classes_))
    top_k = min(3, probabilities.shape[1])
    metrics = {
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "top_1_accuracy": float(accuracy_score(truth, predictions)),
        "top_3_accuracy": float(top_k_accuracy_score(truth, probabilities, k=top_k, labels=labels)),
        "rows": len(truth),
        "classes": len(labels),
    }
    evidence = pd.DataFrame(
        {
            "customer_id": modeling["customer_id"].astype(str),
            "truth_category_id": encoder.inverse_transform(truth),
            "predicted_category_id": encoder.inverse_transform(predictions),
            "top_1_probability": probabilities.max(axis=1),
        }
    )
    return metrics, evidence


def _score_autoencoder(
    test_table: pd.DataFrame, artifact_path: Path, *, threshold: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    model, metadata = load_autoencoder(artifact_path)
    feature_names = list(metadata["feature_names"])
    transform = metadata["transform"]
    values = test_table.loc[:, feature_names].to_numpy(dtype="float64")
    medians = np.asarray(transform["medians"], dtype="float64")
    means = np.asarray(transform["means"], dtype="float64")
    scales = np.asarray(transform["scales"], dtype="float64")
    scaled = ((np.where(np.isnan(values), medians, values) - means) / scales).astype("float32")
    with torch.no_grad():
        tensor = torch.from_numpy(scaled)
        reconstructed = model(tensor).numpy()
    errors = ((reconstructed - scaled) ** 2).mean(axis=1)
    flags = errors >= threshold
    metrics = {
        "threshold": threshold,
        "error_min": float(errors.min()),
        "error_median": float(np.median(errors)),
        "error_mean": float(errors.mean()),
        "error_max": float(errors.max()),
        "flagged_count": int(flags.sum()),
        "flagged_rate": float(flags.mean()),
        "rows": len(errors),
        "interpretation": "manual-review anomaly candidate; not confirmed fraud",
    }
    evidence = pd.DataFrame(
        {
            "customer_id": test_table["customer_id"].astype(str),
            "reconstruction_error": errors,
            "is_anomaly_candidate": flags,
        }
    )
    return metrics, evidence


def run_final_evaluation(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read the final test once, score frozen artifacts, and persist immutable-style evidence."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    freeze_config = config["model_freeze"]
    freeze_path = (
        resolve_project_path(freeze_config["evidence_directory"], project_root=root)
        / "model_freeze.json"
    )
    output = resolve_project_path(freeze_config["final_evaluation_directory"], project_root=root)
    output.mkdir(parents=True, exist_ok=True)
    freeze = _validate_final_lock(root, freeze_path, output)
    freeze_sha256 = _sha256(freeze_path)
    lock = {
        "evaluation_attempt": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": freeze_sha256,
        "status_sentinel": FREEZE_SENTINEL,
        "choices_frozen": True,
    }
    lock_path = output / "final_evaluation_execution_lock.json"
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)

    churn_table = pd.read_parquet(
        resolve_project_path(config["outputs"]["churn_features"], project_root=root)
    )
    clv_table = pd.read_parquet(
        resolve_project_path(config["outputs"]["clv_features"], project_root=root)
    )
    test_churn = churn_table.loc[churn_table["partition"].astype("string").eq("test")].copy()
    test_clv = clv_table.loc[clv_table["partition"].astype("string").eq("test")].copy()
    if test_churn.empty or test_clv.empty:
        raise DataValidationError("Final churn and CLV test partitions must be non-empty")

    churn_bundle = joblib.load(root / freeze["production_churn"]["artifact"])
    churn_features = list(churn_bundle["metadata"]["feature_names"])
    churn_probabilities = churn_bundle["pipeline"].predict_proba(test_churn.loc[:, churn_features])[
        :, 1
    ]
    churn_threshold = float(freeze["production_churn"]["threshold"])
    churn_metrics = classification_metrics_at_threshold(
        test_churn["churn"].astype("int64"),
        churn_probabilities,
        threshold=churn_threshold,
    )
    churn_predictions = pd.DataFrame(
        {
            "customer_id": test_churn["customer_id"].astype(str),
            "truth": test_churn["churn"].astype("int64"),
            "probability": churn_probabilities,
            "prediction": (churn_probabilities >= churn_threshold).astype("int8"),
        }
    )

    clv_bundle = joblib.load(root / freeze["production_clv"]["artifact"])
    clv_features = list(clv_bundle["metadata"]["feature_names"])
    clv_predictions_values = np.maximum(
        clv_bundle["pipeline"].predict(test_clv.loc[:, clv_features]), 0.0
    )
    clv_truth = test_clv["clv_180d_target"].to_numpy(dtype="float64")
    clv_metrics = {
        "mae": float(mean_absolute_error(clv_truth, clv_predictions_values)),
        "rmse": float(mean_squared_error(clv_truth, clv_predictions_values) ** 0.5),
        "r2": float(r2_score(clv_truth, clv_predictions_values)),
        "rows": len(clv_truth),
    }
    clv_predictions = pd.DataFrame(
        {
            "customer_id": test_clv["customer_id"].astype(str),
            "truth_clv_180d": clv_truth,
            "predicted_clv_180d": clv_predictions_values,
        }
    )

    transactions = pd.read_parquet(
        resolve_project_path(config["cleaning"]["interim_transactions"], project_root=root)
    )
    taxonomy = pd.read_parquet(
        resolve_project_path(config["outputs"]["product_taxonomy"], project_root=root)
    )
    split = pd.read_parquet(
        resolve_project_path(config["outputs"]["customer_split"], project_root=root)
    )
    source_metadata = json.loads(
        resolve_project_path(config["outputs"]["step02_metadata"], project_root=root).read_text(
            encoding="utf-8"
        )
    )
    lstm_metrics, lstm_predictions = _score_lstm(
        root,
        config,
        transactions,
        taxonomy,
        split,
        pd.Timestamp(source_metadata["observation_end"]),
        float(freeze["production_next_purchase"]["threshold"]),
    )
    next_category_bundle = joblib.load(root / freeze["production_next_category"]["artifact"])
    next_category_metrics, next_category_predictions = _score_next_category(
        churn_table, transactions, taxonomy, next_category_bundle
    )
    autoencoder_metrics, autoencoder_scores = _score_autoencoder(
        test_churn,
        root / freeze["production_autoencoder"]["artifact"],
        threshold=float(freeze["production_autoencoder"]["threshold"]),
    )

    churn_predictions.to_csv(output / "churn_test_predictions.csv", index=False)
    clv_predictions.to_csv(output / "clv_test_predictions.csv", index=False)
    lstm_predictions.to_csv(output / "lstm_test_predictions.csv", index=False)
    next_category_predictions.to_csv(output / "next_category_test_predictions.csv", index=False)
    autoencoder_scores.to_csv(output / "autoencoder_test_scores.csv", index=False)
    final_metrics: dict[str, Any] = {
        "step": "STEP 06",
        "evaluation_attempt": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": freeze_sha256,
        "freeze_version": freeze["freeze_version"],
        "source_sha256": freeze["source_sha256"],
        "split_version": freeze["split_version"],
        "feature_schema_version": freeze["feature_schema_version"],
        "choices_frozen_before_test": True,
        "held_out_test_accessed": True,
        "held_out_test_evaluations": 1,
        "production_churn_model": freeze["production_churn"]["model"],
        "production_clv_model": freeze["production_clv"]["model"],
        "churn": churn_metrics,
        "clv": clv_metrics,
        "next_purchase_lstm": lstm_metrics,
        "next_category": next_category_metrics,
        "autoencoder": autoencoder_metrics,
        "target_assessment": {
            "churn_roc_auc_ge_0_80": bool(churn_metrics["roc_auc"] >= 0.80),
            "churn_recall_ge_0_70": bool(churn_metrics["recall"] >= 0.70),
            "clv_r2_ge_0_60": bool(clv_metrics["r2"] >= 0.60),
        },
    }
    _write_json(output / "final_metrics.json", final_metrics)
    LOGGER.info(
        "Single final held-out evaluation completed",
        extra={"event": "final_evaluation_completed", "evaluation_attempt": 1},
    )
    return final_metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the final-evaluation command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the single held-out final evaluation."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config["logging"]["level"]))
    run_final_evaluation(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
