"""Frozen-taxonomy next-purchase-category target and LightGBM classifier."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data.validation import DataValidationError
from src.models.common import (
    artifact_metadata,
    log_mlflow_run,
    save_joblib_artifact,
    timed_fit,
)


def build_next_category_targets(
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Find the deterministic dominant category in each customer's next valid invoice."""
    eligible = transactions.loc[
        transactions["invoice_date"].ge(cutoff)
        & transactions["is_positive_purchase"].fillna(False)
        & transactions["is_product"].fillna(False)
        & transactions["customer_id"].notna(),
        [
            "customer_id",
            "invoice",
            "invoice_date",
            "stock_code",
            "quantity",
            "gross_positive_value",
        ],
    ].copy()
    if eligible.empty:
        raise DataValidationError("No valid future invoices exist for next-category targets")
    order_keys = (
        eligible[["customer_id", "invoice", "invoice_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "invoice_date", "invoice"], kind="mergesort")
        .drop_duplicates("customer_id", keep="first")
    )
    next_order = eligible.merge(
        order_keys[["customer_id", "invoice"]],
        on=["customer_id", "invoice"],
        how="inner",
        validate="many_to_one",
    )
    category_lookup = taxonomy[["stock_code", "category_id"]].drop_duplicates("stock_code")
    next_order = next_order.merge(
        category_lookup, on="stock_code", how="left", validate="many_to_one"
    )
    next_order["category_id"] = next_order["category_id"].fillna(-1).astype("int16")
    category_totals = (
        next_order.groupby(["customer_id", "category_id"], observed=True, as_index=False)
        .agg(
            merchandise_value=("gross_positive_value", "sum"),
            merchandise_quantity=("quantity", "sum"),
        )
        .sort_values(
            ["customer_id", "merchandise_value", "merchandise_quantity", "category_id"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    targets = category_totals.drop_duplicates("customer_id", keep="first").rename(
        columns={"category_id": "next_category_id"}
    )
    return targets.reset_index(drop=True)


def _metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: np.ndarray,
) -> dict[str, float]:
    top_k = min(3, probabilities.shape[1])
    return {
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "top_1_accuracy": float(accuracy_score(truth, predictions)),
        "top_3_accuracy": float(top_k_accuracy_score(truth, probabilities, k=top_k, labels=labels)),
    }


def train_next_category_model(
    feature_table: pd.DataFrame,
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    schema_version: str,
    split_version: str,
    source_sha256: str,
    seed: int,
    artifact_directory: Path,
    evidence_directory: Path,
    mlflow_tags: Mapping[str, str],
    search_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Tune multiclass LightGBM on train-only folds and evaluate validation only."""
    cutoff_values = feature_table["cutoff_timestamp"].drop_duplicates()
    if len(cutoff_values) != 1:
        raise DataValidationError("Next-category features must share one cutoff")
    allowed_ids = set(
        feature_table.loc[
            feature_table["partition"].astype("string").isin(["train", "validation"]),
            "customer_id",
        ].astype("string")
    )
    modeling_transactions = transactions.loc[transactions["customer_id"].isin(allowed_ids)]
    targets = build_next_category_targets(
        modeling_transactions,
        taxonomy,
        cutoff=pd.Timestamp(cutoff_values.iloc[0]),
    )
    modeling = feature_table.merge(targets, on="customer_id", how="inner", validate="one_to_one")
    train = modeling.loc[modeling["partition"].astype("string").eq("train")]
    validation = modeling.loc[modeling["partition"].astype("string").eq("validation")]
    if train.empty or validation.empty:
        raise DataValidationError(
            "Next-category train and validation populations must be non-empty"
        )
    encoder = LabelEncoder().fit(train["next_category_id"].astype("int64"))
    unseen = sorted(set(validation["next_category_id"]).difference(encoder.classes_))
    if unseen:
        raise DataValidationError(f"Validation contains unseen next categories: {unseen}")
    y_train = encoder.transform(train["next_category_id"].astype("int64"))
    y_validation = encoder.transform(validation["next_category_id"].astype("int64"))
    x_train = train.loc[:, feature_names]
    x_validation = validation.loc[:, feature_names]
    class_counts = pd.Series(y_train).value_counts()
    rare_classes = int(class_counts.lt(5).sum())

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    objective="multiclass",
                    num_class=len(encoder.classes_),
                    class_weight="balanced",
                    verbosity=-1,
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ]
    )
    parameters: list[dict[str, list[Any]]] = [
        {f"model__{key}": [value] for key, value in candidate.items()}
        for candidate in search_config["candidates"]
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipeline,
        parameters,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        refit="macro_f1",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The least populated class in y has only",
            category=UserWarning,
        )
        fitted, training_seconds = timed_fit(search.fit, x_train, y_train)
    probabilities = fitted.predict_proba(x_validation)
    predictions = fitted.predict(x_validation)
    labels = np.arange(len(encoder.classes_))
    validation_metrics = _metrics(
        y_validation,
        predictions,
        probabilities,
        labels=labels,
    )
    baseline_class = int(pd.Series(y_train).mode().iloc[0])
    baseline_predictions = np.full(len(y_validation), baseline_class, dtype="int64")
    class_priors = np.bincount(y_train, minlength=len(labels)).astype("float64")
    class_priors /= class_priors.sum()
    baseline_probabilities = np.tile(class_priors, (len(y_validation), 1))
    baseline_metrics = _metrics(
        y_validation,
        baseline_predictions,
        baseline_probabilities,
        labels=labels,
    )
    best_index = int(fitted.best_index_)
    cv_metrics = {
        "cv_macro_f1_mean": float(fitted.cv_results_["mean_test_macro_f1"][best_index]),
        "cv_top_1_accuracy_mean": float(fitted.cv_results_["mean_test_accuracy"][best_index]),
    }
    flat_metrics = {
        **cv_metrics,
        **{f"validation_{key}": value for key, value in validation_metrics.items()},
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
    }
    run_id = log_mlflow_run(
        run_name="next_category_lightgbm",
        family="next_category",
        parameters=fitted.best_params_,
        metrics=flat_metrics,
        tags={
            "model_name": "lightgbm_multiclass",
            "rare_training_classes_below_five": str(rare_classes),
            **dict(mlflow_tags),
        },
        training_seconds=training_seconds,
    )
    metadata = artifact_metadata(
        model_family="next_category",
        model_name="lightgbm_multiclass",
        source_sha256=source_sha256,
        feature_schema_version=schema_version,
        split_version=split_version,
        seed=seed,
        feature_names=feature_names,
        parameters=fitted.best_params_,
        metrics={
            "cv": cv_metrics,
            "validation": validation_metrics,
            "baseline": baseline_metrics,
            "mlflow_run_id": run_id,
        },
        training_seconds=training_seconds,
    )
    artifact_path = artifact_directory / "next_category_lightgbm.joblib"
    save_joblib_artifact(
        {
            "pipeline": fitted.best_estimator_,
            "label_encoder": encoder,
            "metadata": metadata,
        },
        artifact_path,
    )
    rows = [
        {
            "model": "most_popular_category_baseline",
            **baseline_metrics,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "classes": len(labels),
            "rare_training_classes_below_five": rare_classes,
            "held_out_test_accessed": False,
        },
        {
            "model": "lightgbm_multiclass",
            **validation_metrics,
            **cv_metrics,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "classes": len(labels),
            "rare_training_classes_below_five": rare_classes,
            "best_parameters": json.dumps(fitted.best_params_, sort_keys=True),
            "training_seconds": training_seconds,
            "mlflow_run_id": run_id,
            "artifact": artifact_path.name,
            "held_out_test_accessed": False,
        },
    ]
    evidence = pd.DataFrame(rows)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(evidence_directory / "next_category_evaluation.csv", index=False)
    return evidence
