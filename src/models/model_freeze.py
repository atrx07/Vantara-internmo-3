"""Validation-only production model selection and serving-artifact freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data.validation import DataValidationError
from src.models.common import load_feature_schema
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def consolidated_churn_comparison(
    classical_path: str | Path, ann_summary_path: str | Path
) -> pd.DataFrame:
    """Generate the seven-model validation comparison directly from tracked evidence."""
    classical = pd.read_csv(classical_path)
    required = {
        "model",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "validation_confusion_matrix",
        "held_out_test_accessed",
    }
    missing = sorted(required.difference(classical.columns))
    if missing:
        raise DataValidationError(f"Classical comparison is missing columns: {missing}")
    rows = [
        {
            "model": str(row["model"]),
            "family": "classical",
            "validation_accuracy": float(row["validation_accuracy"]),
            "validation_precision": float(row["validation_precision"]),
            "validation_recall": float(row["validation_recall"]),
            "validation_f1": float(row["validation_f1"]),
            "validation_roc_auc": float(row["validation_roc_auc"]),
            "validation_confusion_matrix": str(row["validation_confusion_matrix"]),
            "artifact": str(row["artifact"]),
            "held_out_test_accessed": bool(row["held_out_test_accessed"]),
        }
        for _, row in classical.iterrows()
    ]
    ann_summary = json.loads(Path(ann_summary_path).read_text(encoding="utf-8"))
    ann = ann_summary["ann"]
    metrics = ann["metrics"]
    rows.append(
        {
            "model": "ann",
            "family": "deep",
            "validation_accuracy": float(metrics["accuracy"]),
            "validation_precision": float(metrics["precision"]),
            "validation_recall": float(metrics["recall"]),
            "validation_f1": float(metrics["f1"]),
            "validation_roc_auc": float(metrics["roc_auc"]),
            "validation_confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "artifact": str(ann["artifact"]),
            "held_out_test_accessed": bool(ann_summary["held_out_test_accessed"]),
        }
    )
    comparison = pd.DataFrame(rows)
    if comparison["held_out_test_accessed"].any():
        raise DataValidationError("Validation comparison contains test-accessed evidence")
    return comparison.sort_values(
        ["validation_roc_auc", "validation_recall"],
        ascending=[False, False],
        ignore_index=True,
    )


def select_churn_model(
    comparison: pd.DataFrame, *, minimum_recall: float, roc_auc_tolerance: float
) -> pd.Series:
    """Apply the locked recall, ROC-AUC tolerance, then recall tie-break rule."""
    eligible = comparison.loc[comparison["validation_recall"].ge(minimum_recall)].copy()
    if eligible.empty:
        raise DataValidationError("No churn model satisfies the minimum validation recall")
    leading_auc = float(eligible["validation_roc_auc"].max())
    near_best = eligible.loc[eligible["validation_roc_auc"].ge(leading_auc - roc_auc_tolerance)]
    return near_best.sort_values(
        ["validation_recall", "validation_roc_auc", "model"],
        ascending=[False, False, True],
    ).iloc[0]


def select_f2_threshold(
    truth: Sequence[int],
    probabilities: Sequence[float],
    *,
    minimum_recall: float,
    beta: float,
) -> tuple[float, pd.DataFrame]:
    """Choose the validation-only F-beta maximum while preserving minimum recall."""
    labels = np.asarray(truth, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    rows: list[dict[str, float]] = []
    beta_squared = beta**2
    for index, threshold in enumerate(thresholds):
        denominator = beta_squared * precision[index] + recall[index]
        f_beta = (
            (1.0 + beta_squared) * precision[index] * recall[index] / denominator
            if denominator > 0.0
            else 0.0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f_beta": float(f_beta),
            }
        )
    curve = pd.DataFrame(rows)
    eligible = curve.loc[curve["recall"].ge(minimum_recall)]
    if eligible.empty:
        raise DataValidationError("No validation threshold preserves the minimum recall")
    selected = eligible.sort_values(
        ["f_beta", "recall", "precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"]), curve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise DataValidationError(f"Required serving artifact does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def ensure_freeze_is_open(final_evidence: Path) -> None:
    """Refuse to replace a model freeze after final evaluation has started."""
    protected = (
        final_evidence / "final_evaluation_execution_lock.json",
        final_evidence / "final_metrics.json",
    )
    if any(path.exists() for path in protected):
        raise DataValidationError(
            "Model freeze is immutable after final held-out evaluation has started"
        )


def _inventory(root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def run_model_freeze(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze validation-selected models and threshold without reading test rows."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    freeze_config = config["model_freeze"]
    final_evidence = resolve_project_path(
        freeze_config["final_evaluation_directory"], project_root=root
    )
    ensure_freeze_is_open(final_evidence)
    evidence = resolve_project_path(freeze_config["evidence_directory"], project_root=root)
    evidence.mkdir(parents=True, exist_ok=True)
    step04_evidence = root / "reports" / "modeling"
    step05_evidence = root / "reports" / "deep_learning"
    comparison = consolidated_churn_comparison(
        step04_evidence / "churn_model_comparison.csv",
        step05_evidence / "step05_summary.json",
    )
    selected = select_churn_model(
        comparison,
        minimum_recall=float(freeze_config["minimum_recall"]),
        roc_auc_tolerance=float(freeze_config["roc_auc_tolerance"]),
    )
    selected_name = str(selected["model"])
    if selected_name == "ann":
        raise DataValidationError("ANN selection requires a PyTorch serving adapter not yet frozen")
    source_bundle_path = root / "models_artifacts" / "step04" / f"churn_{selected_name}.joblib"
    source_bundle = joblib.load(source_bundle_path)
    feature_names = tuple(str(value) for value in source_bundle["metadata"]["feature_names"])
    churn_table = pd.read_parquet(
        resolve_project_path(config["outputs"]["churn_features"], project_root=root)
    )
    validation = churn_table.loc[churn_table["partition"].astype("string").eq("validation")]
    if validation.empty:
        raise DataValidationError("No validation rows exist for threshold selection")
    probabilities = source_bundle["pipeline"].predict_proba(validation.loc[:, feature_names])[:, 1]
    threshold, threshold_curve = select_f2_threshold(
        validation["churn"].astype("int64"),
        probabilities,
        minimum_recall=float(freeze_config["minimum_recall"]),
        beta=float(freeze_config["threshold_beta"]),
    )
    selected_curve = threshold_curve.iloc[(threshold_curve["threshold"] - threshold).abs().argmin()]

    clv_comparison = pd.read_csv(step04_evidence / "clv_model_comparison.csv")
    selected_clv = clv_comparison.sort_values(
        ["validation_r2", "validation_rmse"], ascending=[False, True]
    ).iloc[0]
    selected_clv_name = str(selected_clv["model"])
    artifact_root = resolve_project_path(freeze_config["serving_artifact_root"], project_root=root)
    serving_paths = {
        "churn": artifact_root / "churn" / "production_churn.joblib",
        "clv": artifact_root / "clv" / "production_clv.joblib",
        "next_purchase": artifact_root / "next_purchase" / "purchase_lstm.pt",
        "next_category": artifact_root / "next_category" / "next_category_lightgbm.joblib",
        "autoencoder": artifact_root / "autoencoder" / "behavioral_autoencoder.pt",
        "segmentation": artifact_root / "segmentation" / "segmentation_bundle.joblib",
        "recommendation": artifact_root / "recommendation" / "item_to_item_recommender.joblib",
        "product_taxonomy": artifact_root / "product_taxonomy" / "product_taxonomy.parquet",
    }
    production_bundle = {
        "pipeline": source_bundle["pipeline"],
        "metadata": {
            **source_bundle["metadata"],
            "model_version": "vantara-churn-production-v1",
            "selected_threshold": threshold,
            "threshold_method": f"validation_f{float(freeze_config['threshold_beta']):g}_maximum",
            "selection_rule": "recall>=0.70; within 0.01 ROC-AUC prefer recall",
            "validation_selection_metrics": {
                "roc_auc": float(selected["validation_roc_auc"]),
                "recall_at_0_5": float(selected["validation_recall"]),
                "threshold_precision": float(selected_curve["precision"]),
                "threshold_recall": float(selected_curve["recall"]),
                "threshold_f_beta": float(selected_curve["f_beta"]),
            },
            "choices_frozen": True,
            "held_out_test_accessed": False,
        },
    }
    serving_paths["churn"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_bundle, serving_paths["churn"])
    _copy(
        root / "models_artifacts" / "step04" / f"clv_{selected_clv_name}.joblib",
        serving_paths["clv"],
    )
    _copy(root / "models_artifacts" / "step05" / "purchase_lstm.pt", serving_paths["next_purchase"])
    _copy(
        root / "models_artifacts" / "step04" / "next_category_lightgbm.joblib",
        serving_paths["next_category"],
    )
    _copy(
        root / "models_artifacts" / "step05" / "behavioral_autoencoder.pt",
        serving_paths["autoencoder"],
    )
    _copy(
        root / "models_artifacts" / "step04" / "segmentation_bundle.joblib",
        serving_paths["segmentation"],
    )
    _copy(
        root / "models_artifacts" / "step04" / "item_to_item_recommender.joblib",
        serving_paths["recommendation"],
    )
    _copy(
        resolve_project_path(config["outputs"]["product_taxonomy"], project_root=root),
        serving_paths["product_taxonomy"],
    )
    comparison["selected_production"] = comparison["model"].eq(selected_name)
    comparison.to_csv(evidence / "consolidated_churn_validation.csv", index=False)
    threshold_curve.to_csv(evidence / "churn_threshold_curve.csv", index=False)
    schema = load_feature_schema(
        resolve_project_path(config["analysis"]["churn_feature_schema"], project_root=root)
    )
    inventory = _inventory(root, list(serving_paths.values()))
    freeze_record: dict[str, Any] = {
        "freeze_version": "vantara-model-freeze-v1",
        "step": "STEP 06",
        "owner_step06_approval_received": True,
        "choices_frozen": True,
        "final_test_status": "NOT_RUN",
        "held_out_test_accessed": False,
        "source_sha256": source_bundle["metadata"]["source_sha256"],
        "split_version": source_bundle["metadata"]["split_version"],
        "feature_schema_version": schema["schema_version"],
        "feature_count": schema["feature_count"],
        "production_churn": {
            "model": selected_name,
            "artifact": serving_paths["churn"].relative_to(root).as_posix(),
            "threshold": threshold,
            "threshold_beta": float(freeze_config["threshold_beta"]),
            "validation_roc_auc": float(selected["validation_roc_auc"]),
            "validation_recall_at_0_5": float(selected["validation_recall"]),
            "validation_threshold_precision": float(selected_curve["precision"]),
            "validation_threshold_recall": float(selected_curve["recall"]),
            "validation_threshold_f_beta": float(selected_curve["f_beta"]),
        },
        "production_clv": {
            "model": selected_clv_name,
            "artifact": serving_paths["clv"].relative_to(root).as_posix(),
            "validation_r2": float(selected_clv["validation_r2"]),
            "validation_rmse": float(selected_clv["validation_rmse"]),
            "cv_r2_mean": float(selected_clv["cv_r2_mean"]),
        },
        "production_next_purchase": {
            "model": "purchase_lstm",
            "artifact": serving_paths["next_purchase"].relative_to(root).as_posix(),
            "threshold": 0.5,
        },
        "production_next_category": {
            "model": "lightgbm_multiclass",
            "artifact": serving_paths["next_category"].relative_to(root).as_posix(),
        },
        "production_autoencoder": {
            "model": "behavioral_autoencoder",
            "artifact": serving_paths["autoencoder"].relative_to(root).as_posix(),
            "threshold": 0.1707874834537506,
            "threshold_method": "validation_reconstruction_error_percentile_99",
        },
        "serving_artifacts": inventory,
    }
    freeze_path = evidence / "model_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze_record, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    LOGGER.info(
        "Validation-only model freeze completed",
        extra={"event": "model_freeze_completed", "model": selected_name, "threshold": threshold},
    )
    return freeze_record


def build_parser() -> argparse.ArgumentParser:
    """Build the model-freeze command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation-only production model freezing."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config["logging"]["level"]))
    run_model_freeze(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
