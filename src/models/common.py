"""Shared STEP 04 model-data, metric, artifact, and MLflow utilities."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import joblib
import mlflow
import numpy as np
import pandas as pd
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
)

from src.data.validation import DataValidationError

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class SupervisedPartitions:
    """Training and validation matrices that deliberately omit the held-out test."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    train_customer_ids: tuple[str, ...]
    validation_customer_ids: tuple[str, ...]


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    """Load and validate the canonical frozen churn feature schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("selected_features")
    if not isinstance(features, list) or not features:
        raise DataValidationError("Frozen feature schema has no selected_features")
    if int(payload.get("feature_count", -1)) != len(features):
        raise DataValidationError("Frozen feature schema count does not match its feature list")
    return payload


def supervised_partitions(
    table: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    target_name: str,
) -> SupervisedPartitions:
    """Select only training and validation rows under the immutable split contract."""
    required = {"customer_id", "partition", target_name, *feature_names}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise DataValidationError(f"Modeling table is missing required columns: {missing}")
    partition = table["partition"].astype("string")
    unexpected = sorted(
        set(partition.dropna().unique()).difference({"train", "validation", "test"})
    )
    if unexpected:
        raise DataValidationError(f"Unexpected partition labels: {unexpected}")
    train = table.loc[partition.eq("train")]
    validation = table.loc[partition.eq("validation")]
    if train.empty or validation.empty:
        raise DataValidationError("Training and validation partitions must both be non-empty")
    return SupervisedPartitions(
        x_train=train.loc[:, feature_names].copy(),
        y_train=train[target_name].copy(),
        x_validation=validation.loc[:, feature_names].copy(),
        y_validation=validation[target_name].copy(),
        train_customer_ids=tuple(train["customer_id"].astype("string")),
        validation_customer_ids=tuple(validation["customer_id"].astype("string")),
    )


def classification_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    """Calculate the governed binary-classification metrics at threshold 0.5."""
    truth = np.asarray(y_true, dtype="int64")
    scores = np.asarray(probabilities, dtype="float64")
    predictions = scores >= 0.5
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "confusion_matrix": matrix.astype("int64").tolist(),
        "threshold": 0.5,
    }


def regression_metrics(y_true: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    """Calculate business-scale CLV regression metrics."""
    truth = np.asarray(y_true, dtype="float64")
    predicted = np.asarray(predictions, dtype="float64")
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(mean_squared_error(truth, predicted) ** 0.5),
        "r2": float(r2_score(truth, predicted)),
    }


def save_joblib_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    """Persist one model bundle to its canonical artifact path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(payload), destination)


def artifact_metadata(
    *,
    model_family: str,
    model_name: str,
    source_sha256: str,
    feature_schema_version: str,
    split_version: str,
    seed: int,
    feature_names: Sequence[str],
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    """Build the compatibility and experiment metadata saved with each model."""
    return {
        "model_family": model_family,
        "model_name": model_name,
        "source_sha256": source_sha256,
        "feature_schema_version": feature_schema_version,
        "split_version": split_version,
        "seed": seed,
        "feature_names": list(feature_names),
        "parameters": _json_safe(parameters),
        "metrics": _json_safe(metrics),
        "training_seconds": float(training_seconds),
        "held_out_test_accessed": False,
    }


def configure_mlflow(tracking_directory: str | Path, experiment_name: str) -> None:
    """Configure the ignored local MLflow file store and named experiment."""
    directory = Path(tracking_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(directory.as_uri())
    mlflow.set_experiment(experiment_name)


def log_mlflow_run(
    *,
    run_name: str,
    family: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    training_seconds: float,
) -> str:
    """Log one auditable local MLflow run and return its run ID."""
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({"model_family": family, **dict(tags)})
        safe_parameters = {key: str(value) for key, value in parameters.items()}
        mlflow.log_params(safe_parameters)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.log_metric("training_seconds", float(training_seconds))
        return active.info.run_id


def timed_fit(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
    """Run a fitting callable and return its result with elapsed wall time."""
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
