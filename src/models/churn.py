"""Training-only CV and validation evaluation for the six classical churn models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.common import (
    artifact_metadata,
    classification_metrics,
    log_mlflow_run,
    save_joblib_artifact,
    supervised_partitions,
    timed_fit,
)


def _pipeline(model: BaseEstimator, *, scaled: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _candidate_grid(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, list[Any]]]:
    return [
        {f"model__{key}": [value] for key, value in candidate.items()} for candidate in candidates
    ]


def _model_searches(
    seed: int,
    positive_weight: float,
    config: Mapping[str, Any],
) -> dict[str, tuple[Pipeline, Any]]:
    searches: dict[str, tuple[Pipeline, Any]] = {
        "logistic_regression": (
            _pipeline(
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
                scaled=True,
            ),
            {
                "model__penalty": list(config["logistic_regression"]["penalty"]),
                "model__C": list(config["logistic_regression"]["C"]),
            },
        ),
        "decision_tree": (
            _pipeline(
                DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                scaled=False,
            ),
            _candidate_grid(config["decision_tree"]["candidates"]),
        ),
        "random_forest": (
            _pipeline(
                RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=seed,
                ),
                scaled=False,
            ),
            _candidate_grid(config["random_forest"]["candidates"]),
        ),
        "xgboost": (
            _pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=seed,
                    scale_pos_weight=positive_weight,
                ),
                scaled=False,
            ),
            _candidate_grid(config["xgboost"]["candidates"]),
        ),
        "lightgbm": (
            _pipeline(
                LGBMClassifier(
                    objective="binary",
                    class_weight="balanced",
                    verbosity=-1,
                    n_jobs=1,
                    random_state=seed,
                ),
                scaled=False,
            ),
            _candidate_grid(config["lightgbm"]["candidates"]),
        ),
        "svm": (
            _pipeline(
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
                scaled=True,
            ),
            _candidate_grid(config["svm"]["candidates"]),
        ),
    }
    return searches


def train_churn_models(
    table: pd.DataFrame,
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
    """Tune six required models on training CV and evaluate validation only."""
    partitions = supervised_partitions(
        table,
        feature_names=feature_names,
        target_name="churn",
    )
    y_train = partitions.y_train.astype("int64")
    negatives = max(int(y_train.eq(0).sum()), 1)
    positives = max(int(y_train.eq(1).sum()), 1)
    positive_weight = negatives / positives
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    records: list[dict[str, Any]] = []
    evidence_directory.mkdir(parents=True, exist_ok=True)

    for name, (pipeline, parameters) in _model_searches(
        seed, positive_weight, search_config
    ).items():
        search = GridSearchCV(
            pipeline,
            parameters,
            scoring={"roc_auc": "roc_auc", "recall": "recall", "f1": "f1"},
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        fitted, training_seconds = timed_fit(search.fit, partitions.x_train, y_train)
        probabilities = fitted.predict_proba(partitions.x_validation)[:, 1]
        validation = classification_metrics(partitions.y_validation.astype("int64"), probabilities)
        best_index = int(fitted.best_index_)
        cv_metrics = {
            "cv_roc_auc_mean": float(fitted.cv_results_["mean_test_roc_auc"][best_index]),
            "cv_roc_auc_std": float(fitted.cv_results_["std_test_roc_auc"][best_index]),
            "cv_recall_mean": float(fitted.cv_results_["mean_test_recall"][best_index]),
            "cv_f1_mean": float(fitted.cv_results_["mean_test_f1"][best_index]),
        }
        flat_metrics = {
            **cv_metrics,
            **{
                f"validation_{key}": value
                for key, value in validation.items()
                if key != "confusion_matrix"
            },
        }
        run_id = log_mlflow_run(
            run_name=f"churn_{name}",
            family="churn",
            parameters=fitted.best_params_,
            metrics=flat_metrics,
            tags={"model_name": name, **dict(mlflow_tags)},
            training_seconds=training_seconds,
        )
        metadata = artifact_metadata(
            model_family="churn",
            model_name=name,
            source_sha256=source_sha256,
            feature_schema_version=schema_version,
            split_version=split_version,
            seed=seed,
            feature_names=feature_names,
            parameters=fitted.best_params_,
            metrics={"cv": cv_metrics, "validation": validation, "mlflow_run_id": run_id},
            training_seconds=training_seconds,
        )
        artifact_path = artifact_directory / f"churn_{name}.joblib"
        save_joblib_artifact(
            {"pipeline": fitted.best_estimator_, "metadata": metadata}, artifact_path
        )
        confusion_path = evidence_directory / f"churn_{name}_confusion_matrix.json"
        confusion_path.write_text(
            json.dumps({"labels": [0, 1], "matrix": validation["confusion_matrix"]}, indent=2),
            encoding="utf-8",
        )
        records.append(
            {
                "model": name,
                **cv_metrics,
                **{
                    f"validation_{key}": value
                    for key, value in validation.items()
                    if key != "confusion_matrix"
                },
                "validation_confusion_matrix": json.dumps(validation["confusion_matrix"]),
                "best_parameters": json.dumps(fitted.best_params_, sort_keys=True),
                "training_seconds": training_seconds,
                "mlflow_run_id": run_id,
                "artifact": artifact_path.name,
                "held_out_test_accessed": False,
            }
        )

    comparison = pd.DataFrame(records).sort_values(
        ["validation_roc_auc", "validation_recall"], ascending=[False, False], ignore_index=True
    )
    comparison.to_csv(evidence_directory / "churn_model_comparison.csv", index=False)
    return comparison
