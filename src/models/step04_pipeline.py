"""Execute governed STEP 04 classical modeling and product intelligence."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data.validation import DataValidationError
from src.models.churn import train_churn_models
from src.models.clv import train_clv_models
from src.models.common import configure_mlflow, load_feature_schema
from src.models.next_category import train_next_category_model
from src.recommendation.item_to_item import train_recommender
from src.segmentation.customer_segments import train_segmentation_models
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"Required STEP 04 input does not exist: {path}")
    return pd.read_parquet(path)


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].astype("string").dropna().unique()
    if len(values) != 1:
        raise DataValidationError(f"Expected exactly one {column}, got {list(values)}")
    return str(values[0])


def _run_ids(*frames: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for frame in frames:
        if "mlflow_run_id" in frame:
            values.extend(str(value) for value in frame["mlflow_run_id"].dropna())
    return sorted(set(values))


def _export_mlflow_runs(run_ids: Sequence[str], path: Path) -> pd.DataFrame:
    client = MlflowClient()
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _artifact_reload_smoke(
    artifact_directory: Path,
    churn_table: pd.DataFrame,
    clv_table: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    validation_churn = churn_table.loc[
        churn_table["partition"].astype("string").eq("validation"), feature_names
    ].head(1)
    validation_clv = clv_table.loc[
        clv_table["partition"].astype("string").eq("validation"), feature_names
    ].head(1)
    checks: list[dict[str, Any]] = []
    for path in sorted(artifact_directory.glob("churn_*.joblib")):
        bundle = joblib.load(path)
        probability = float(bundle["pipeline"].predict_proba(validation_churn)[0, 1])
        checks.append({"artifact": path.name, "reload": "PASS", "sample_output": probability})
    for path in sorted(artifact_directory.glob("clv_*.joblib")):
        bundle = joblib.load(path)
        prediction = float(bundle["pipeline"].predict(validation_clv)[0])
        checks.append({"artifact": path.name, "reload": "PASS", "sample_output": prediction})
    category_path = artifact_directory / "next_category_lightgbm.joblib"
    category_bundle = joblib.load(category_path)
    category = int(category_bundle["pipeline"].predict(validation_churn)[0])
    checks.append({"artifact": category_path.name, "reload": "PASS", "sample_output": category})
    segmentation_path = artifact_directory / "segmentation_bundle.joblib"
    segmentation = joblib.load(segmentation_path)
    segment_input = churn_table.loc[
        churn_table["partition"].astype("string").eq("validation"),
        segmentation["feature_names"],
    ].head(1)
    scaled = segmentation["preprocessor"].transform(segment_input)
    segment = int(segmentation["kmeans"].predict(scaled)[0])
    checks.append({"artifact": segmentation_path.name, "reload": "PASS", "sample_output": segment})
    recommender_path = artifact_directory / "item_to_item_recommender.joblib"
    recommender = joblib.load(recommender_path)
    if recommender["neighbor_indices"].shape != recommender["similarities"].shape:
        raise DataValidationError("Reloaded recommender neighbor arrays are incompatible")
    checks.append(
        {
            "artifact": recommender_path.name,
            "reload": "PASS",
            "sample_output": int(recommender["neighbor_indices"].shape[0]),
        }
    )
    return checks


def run_step04(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run all STEP 04 training against train/validation partitions and never final test."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    seed = int(config["project"]["random_seed"])
    modeling = config["modeling"]
    artifact_directory = resolve_project_path(modeling["artifact_directory"], project_root=root)
    evidence_directory = resolve_project_path(modeling["evidence_directory"], project_root=root)
    tracking_directory = resolve_project_path(modeling["tracking_directory"], project_root=root)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    configure_mlflow(tracking_directory, str(modeling["experiment_name"]))

    output_paths = {
        key: resolve_project_path(value, project_root=root)
        for key, value in config["outputs"].items()
    }
    churn_table = _load_frame(output_paths["churn_features"])
    clv_table = _load_frame(output_paths["clv_features"])
    transactions = _load_frame(
        resolve_project_path(config["cleaning"]["interim_transactions"], project_root=root)
    )
    taxonomy = _load_frame(output_paths["product_taxonomy"])
    metadata = json.loads(output_paths["step02_metadata"].read_text(encoding="utf-8"))
    source_sha256 = str(metadata["source_sha256"])
    split_version = _single_value(churn_table, "split_version")
    schema = load_feature_schema(
        resolve_project_path(config["analysis"]["churn_feature_schema"], project_root=root)
    )
    feature_names = tuple(str(value) for value in schema["selected_features"])
    schema_version = str(schema["schema_version"])
    tags = {
        "source_sha256": source_sha256,
        "feature_schema_version": schema_version,
        "split_version": split_version,
        "seed": str(seed),
        "held_out_test_accessed": "false",
    }

    churn = train_churn_models(
        churn_table,
        feature_names=feature_names,
        schema_version=schema_version,
        split_version=split_version,
        source_sha256=source_sha256,
        seed=seed,
        artifact_directory=artifact_directory,
        evidence_directory=evidence_directory,
        mlflow_tags=tags,
        search_config=modeling["churn"],
    )
    clv = train_clv_models(
        clv_table,
        feature_names=feature_names,
        schema_version=schema_version,
        split_version=split_version,
        source_sha256=source_sha256,
        seed=seed,
        artifact_directory=artifact_directory,
        evidence_directory=evidence_directory,
        mlflow_tags=tags,
        search_config=modeling["clv"],
    )
    segmentation_config = modeling["segmentation"]
    segmentation, profiles, assignments = train_segmentation_models(
        churn_table,
        feature_names=tuple(segmentation_config["features"]),
        kmeans_candidates=tuple(int(value) for value in segmentation_config["kmeans_candidates"]),
        gmm_candidates=tuple(int(value) for value in segmentation_config["gmm_candidates"]),
        split_version=split_version,
        source_sha256=source_sha256,
        seed=seed,
        artifact_directory=artifact_directory,
        evidence_directory=evidence_directory,
        mlflow_tags=tags,
    )
    next_category = train_next_category_model(
        churn_table,
        transactions,
        taxonomy,
        feature_names=feature_names,
        schema_version=schema_version,
        split_version=split_version,
        source_sha256=source_sha256,
        seed=seed,
        artifact_directory=artifact_directory,
        evidence_directory=evidence_directory,
        mlflow_tags=tags,
        search_config=modeling["next_category"],
    )
    recommender_config = modeling["recommender"]
    recommender = train_recommender(
        transactions,
        churn_table,
        assignments,
        source_sha256=source_sha256,
        split_version=split_version,
        seed=seed,
        neighbors=int(recommender_config["neighbors"]),
        top_k=int(recommender_config["top_k"]),
        artifact_directory=artifact_directory,
        evidence_directory=evidence_directory,
        mlflow_tags=tags,
    )

    run_ids = _run_ids(churn, clv, segmentation, next_category, recommender)
    mlflow_runs = _export_mlflow_runs(run_ids, evidence_directory / "mlflow_run_summary.csv")
    reload_checks = _artifact_reload_smoke(
        artifact_directory,
        churn_table,
        clv_table,
        feature_names,
    )
    reload_path = evidence_directory / "artifact_reload_smoke.json"
    reload_path.write_text(json.dumps(reload_checks, indent=2), encoding="utf-8")

    best_churn = churn.iloc[0]
    best_clv = clv.iloc[0]
    selected_kmeans = (
        segmentation.loc[segmentation["algorithm"].eq("kmeans")]
        .sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
        .iloc[0]
    )
    selected_gmm = (
        segmentation.loc[segmentation["algorithm"].eq("gmm")]
        .sort_values(["bic", "silhouette"], ascending=[True, False])
        .iloc[0]
    )
    summary: dict[str, Any] = {
        "step": "STEP 04",
        "source_sha256": source_sha256,
        "feature_schema_version": schema_version,
        "feature_count": len(feature_names),
        "split_version": split_version,
        "seed": seed,
        "held_out_test_accessed": False,
        "churn_models": list(churn["model"]),
        "best_validation_churn": {
            "model": str(best_churn["model"]),
            "roc_auc": float(best_churn["validation_roc_auc"]),
            "recall": float(best_churn["validation_recall"]),
        },
        "best_validation_clv": {
            "model": str(best_clv["model"]),
            "r2": float(best_clv["validation_r2"]),
            "rmse": float(best_clv["validation_rmse"]),
        },
        "selected_kmeans_components": int(selected_kmeans["components"]),
        "selected_gmm_components": int(selected_gmm["components"]),
        "segment_profile_rows": len(profiles),
        "next_category": json.loads(next_category.to_json(orient="records")),
        "recommender": json.loads(recommender.to_json(orient="records")),
        "mlflow_run_count": len(mlflow_runs),
        "artifact_reload_checks": reload_checks,
    }
    summary_path = evidence_directory / "step04_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False),
        encoding="utf-8",
    )
    LOGGER.info(
        "STEP 04 modeling pipeline passed",
        extra={"event": "step04_pipeline_passed", "mlflow_runs": len(mlflow_runs)},
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the STEP 04 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute STEP 04 with structured success or failure logging."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    try:
        run_step04(args.config)
    except (DataValidationError, OSError, ValueError, KeyError, mlflow.MlflowException) as exc:
        LOGGER.error(
            "STEP 04 modeling pipeline failed",
            extra={"event": "step04_pipeline_failed"},
            exc_info=exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
