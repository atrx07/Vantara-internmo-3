"""Execute the governed STEP 02 cleaning, snapshot, and feature pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.cleaning import clean_transactions, fit_outlier_thresholds
from src.data.loader import load_transactions
from src.data.snapshots import (
    churn_labels,
    clv_targets,
    derive_canonical_cutoffs,
    eligible_customer_ids,
)
from src.data.splits import create_customer_split, customer_ids_for_partition
from src.data.validation import DataValidationError
from src.features.customer_features import FEATURE_JUSTIFICATIONS, build_customer_features
from src.features.preprocessing import fit_preprocessing_contracts
from src.features.product_artifacts import fit_product_artifacts
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    stable = frame.copy()
    for column in stable.select_dtypes(include=["category"]).columns:
        stable[column] = stable[column].astype("string")
    hashes = pd.util.hash_pandas_object(stable, index=True, categorize=True)
    return hashlib.sha256(hashes.to_numpy().tobytes()).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


def run_step02(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run STEP 02 from immutable raw data and return deterministic audit metadata."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    raw, ingestion = load_transactions(config_path, project_root=root)
    cleaning_config = config["cleaning"]
    snapshot_config = config["snapshots"]
    feature_config = config["features"]
    split_config = feature_config["split"]
    seed = int(split_config["seed"])

    preclean, _ = clean_transactions(raw, cleaning_config=cleaning_config)
    cutoffs = derive_canonical_cutoffs(
        preclean,
        churn_horizon_days=int(snapshot_config["churn_horizon_days"]),
        clv_horizon_days=int(snapshot_config["clv_horizon_days"]),
    )
    customers = eligible_customer_ids(preclean, cutoff=cutoffs.clv_cutoff)
    split = create_customer_split(
        customers,
        train_fraction=float(split_config["train_fraction"]),
        validation_fraction=float(split_config["validation_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        seed=seed,
        version=str(split_config["version"]),
    )
    training_ids = customer_ids_for_partition(split, "train")
    threshold_history = preclean.loc[preclean["invoice_date"].lt(cutoffs.clv_cutoff)]
    thresholds = fit_outlier_thresholds(
        threshold_history,
        training_customer_ids=training_ids,
        config=cleaning_config["outliers"],
    )
    cleaned, cleaning_summary = clean_transactions(
        raw,
        cleaning_config=cleaning_config,
        outlier_thresholds=thresholds,
    )
    if eligible_customer_ids(cleaned, cutoff=cutoffs.clv_cutoff) != customers:
        raise DataValidationError("Outlier audit unexpectedly changed the modeling population")

    artifacts = fit_product_artifacts(
        cleaned,
        training_customer_ids=training_ids,
        cutoff=cutoffs.clv_cutoff,
        feature_config=feature_config,
        seed=seed,
    )
    common_feature_args = {
        "transactions": cleaned,
        "customer_ids": customers,
        "artifacts": artifacts,
        "training_customer_ids": training_ids,
        "trend_window_days": int(snapshot_config["trend_window_days"]),
        "markdown_price_ratio": float(feature_config["markdown_price_ratio"]),
    }
    churn_features, churn_engagement = build_customer_features(
        cutoff=cutoffs.churn_cutoff, **common_feature_args
    )
    clv_features, clv_engagement = build_customer_features(
        cutoff=cutoffs.clv_cutoff, **common_feature_args
    )
    churn = churn_labels(
        cleaned,
        customer_ids=customers,
        cutoff=cutoffs.churn_cutoff,
        horizon_days=int(snapshot_config["churn_horizon_days"]),
    )
    clv = clv_targets(
        cleaned,
        customer_ids=customers,
        cutoff=cutoffs.clv_cutoff,
        horizon_days=int(snapshot_config["clv_horizon_days"]),
    )
    partition_columns = split[["customer_id", "partition", "split_version"]]
    churn_table = churn_features.merge(churn, on="customer_id", validate="one_to_one")
    churn_table = churn_table.merge(partition_columns, on="customer_id", validate="one_to_one")
    clv_table = clv_features.merge(clv, on="customer_id", validate="one_to_one")
    clv_table = clv_table.merge(partition_columns, on="customer_id", validate="one_to_one")

    preprocessors = {
        "churn": fit_preprocessing_contracts(churn_table, split, excluded_columns={"churn"}),
        "clv": fit_preprocessing_contracts(clv_table, split, excluded_columns={"clv_180d_target"}),
        "engagement": {"churn": churn_engagement, "clv": clv_engagement},
    }

    output_paths = {
        name: resolve_project_path(value, project_root=root)
        for name, value in config["outputs"].items()
    }
    interim_path = resolve_project_path(cleaning_config["interim_transactions"], project_root=root)
    outlier_path = resolve_project_path(cleaning_config["outlier_audit"], project_root=root)
    _write_parquet(cleaned, interim_path)
    _write_parquet(cleaned.loc[cleaned["is_statistical_outlier"]], outlier_path)
    _write_parquet(split, output_paths["customer_split"])
    _write_parquet(churn_table, output_paths["churn_features"])
    _write_parquet(clv_table, output_paths["clv_features"])
    _write_parquet(artifacts.taxonomy, output_paths["product_taxonomy"])
    _write_parquet(artifacts.reference_prices, output_paths["product_reference_prices"])
    _write_parquet(artifacts.popularity, output_paths["product_frequency_encoding"])
    output_paths["preprocessing_artifacts"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessors, output_paths["preprocessing_artifacts"])

    split_counts = {
        str(key): int(value) for key, value in split["partition"].value_counts().items()
    }
    metadata: dict[str, Any] = {
        "step": "STEP 02",
        "source_sha256": ingestion.sha256,
        "observation_end": cutoffs.observation_end.isoformat(sep=" "),
        "churn_cutoff": cutoffs.churn_cutoff.isoformat(sep=" "),
        "clv_cutoff": cutoffs.clv_cutoff.isoformat(sep=" "),
        "cleaning_summary": asdict(cleaning_summary),
        "outlier_thresholds": asdict(thresholds),
        "customer_count": len(customers),
        "split_counts": split_counts,
        "split_seed": seed,
        "split_version": str(split_config["version"]),
        "taxonomy_version": artifacts.taxonomy_version,
        "taxonomy_selected_clusters": artifacts.selected_clusters,
        "taxonomy_candidates": artifacts.taxonomy_candidates.to_dict(orient="records"),
        "taxonomy_top_terms": {str(key): list(value) for key, value in artifacts.top_terms.items()},
        "reference_price_products": len(artifacts.reference_prices),
        "frequency_encoded_products": len(artifacts.popularity),
        "churn_rows": len(churn_table),
        "churn_rate": float(churn_table["churn"].mean()),
        "clv_rows": len(clv_table),
        "clv_positive_rate": float(clv_table["clv_180d_target"].gt(0).mean()),
        "feature_justifications": FEATURE_JUSTIFICATIONS,
        "fingerprints": {
            "cleaned_transactions": _frame_fingerprint(cleaned),
            "customer_split": _frame_fingerprint(split),
            "churn_features": _frame_fingerprint(churn_table),
            "clv_features": _frame_fingerprint(clv_table),
            "product_taxonomy": _frame_fingerprint(artifacts.taxonomy),
        },
    }
    metadata_path = output_paths["step02_metadata"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "STEP 02 data pipeline passed",
        extra={"event": "step02_pipeline_passed", "rows": len(cleaned)},
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the STEP 02 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run STEP 02 with structured success/failure logging."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    try:
        run_step02(args.config)
    except (DataValidationError, OSError, ValueError, KeyError) as exc:
        LOGGER.error(
            "STEP 02 data pipeline failed",
            extra={"event": "step02_pipeline_failed"},
            exc_info=exc,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
