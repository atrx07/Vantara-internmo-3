"""Training-only CLV regression CV with Ridge and XGBRegressor candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.validation import DataValidationError
from src.models.common import (
    artifact_metadata,
    log_mlflow_run,
    regression_metrics,
    save_joblib_artifact,
    supervised_partitions,
    timed_fit,
)


class TrainingTargetRangeRegressor(RegressorMixin, BaseEstimator):
    """Fit a regressor and cap business predictions to its training-target range."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        target: pd.Series | np.ndarray,
    ) -> TrainingTargetRangeRegressor:
        """Fit a clone and learn the nonnegative prediction bounds from training only."""
        values = np.asarray(target, dtype="float64")
        self.target_min_ = 0.0
        self.target_max_ = float(np.max(values))
        self.estimator_ = clone(self.estimator).fit(features, values)
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predictions clipped to the fitted training-target range."""
        predictions = np.asarray(self.estimator_.predict(features), dtype="float64")
        return np.clip(predictions, self.target_min_, self.target_max_)


def stratified_clv_folds(target: pd.Series, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create five training-only folds balanced on quantiles of log1p CLV."""
    logged = np.log1p(target.astype("float64"))
    bins = pd.qcut(logged.rank(method="first"), q=10, labels=False, duplicates="drop")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(target)), bins))


def _candidate_grid(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, list[Any]]]:
    return [
        {f"estimator__regressor__model__{key}": [value] for key, value in candidate.items()}
        for candidate in candidates
    ]


def _regression_searches(
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, tuple[TransformedTargetRegressor, Any]]:
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    xgboost = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {
        "ridge": (
            TrainingTargetRangeRegressor(
                TransformedTargetRegressor(
                    regressor=ridge,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=True,
                )
            ),
            {"estimator__regressor__model__alpha": list(config["ridge_alpha"])},
        ),
        "xgboost_regressor": (
            TrainingTargetRangeRegressor(
                TransformedTargetRegressor(
                    regressor=xgboost,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=True,
                )
            ),
            _candidate_grid(config["xgboost_candidates"]),
        ),
    }


def train_clv_models(
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
    """Tune Ridge and XGBRegressor using train-only CV and validation-only metrics."""
    if search_config.get("prediction_cap") != "training_target_max":
        raise DataValidationError(
            "STEP 04 CLV prediction_cap must be the governed training_target_max method"
        )
    partitions = supervised_partitions(
        table,
        feature_names=feature_names,
        target_name="clv_180d_target",
    )
    y_train = partitions.y_train.astype("float64")
    folds = stratified_clv_folds(y_train, seed=seed)
    records: list[dict[str, Any]] = []
    evidence_directory.mkdir(parents=True, exist_ok=True)

    for name, (estimator, parameters) in _regression_searches(seed, search_config).items():
        search = GridSearchCV(
            estimator,
            parameters,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error",
            },
            refit="neg_rmse",
            cv=folds,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        fitted, training_seconds = timed_fit(search.fit, partitions.x_train, y_train)
        predictions = np.maximum(fitted.predict(partitions.x_validation), 0.0)
        validation = regression_metrics(partitions.y_validation.astype("float64"), predictions)
        best_index = int(fitted.best_index_)
        cv_metrics = {
            "cv_r2_mean": float(fitted.cv_results_["mean_test_r2"][best_index]),
            "cv_mae_mean": float(-fitted.cv_results_["mean_test_neg_mae"][best_index]),
            "cv_rmse_mean": float(-fitted.cv_results_["mean_test_neg_rmse"][best_index]),
        }
        flat_metrics = {
            **cv_metrics,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        run_id = log_mlflow_run(
            run_name=f"clv_{name}",
            family="clv",
            parameters=fitted.best_params_,
            metrics=flat_metrics,
            tags={"model_name": name, "target_transform": "log1p", **dict(mlflow_tags)},
            training_seconds=training_seconds,
        )
        metadata = artifact_metadata(
            model_family="clv",
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
        artifact_path = artifact_directory / f"clv_{name}.joblib"
        save_joblib_artifact(
            {"pipeline": fitted.best_estimator_, "metadata": metadata}, artifact_path
        )
        records.append(
            {
                "model": name,
                **cv_metrics,
                **{f"validation_{key}": value for key, value in validation.items()},
                "best_parameters": json.dumps(fitted.best_params_, sort_keys=True),
                "training_seconds": training_seconds,
                "mlflow_run_id": run_id,
                "artifact": artifact_path.name,
                "target_transform": "log1p",
                "held_out_test_accessed": False,
            }
        )

    comparison = pd.DataFrame(records).sort_values(
        ["validation_r2", "validation_rmse"], ascending=[False, True], ignore_index=True
    )
    comparison.to_csv(evidence_directory / "clv_model_comparison.csv", index=False)
    return comparison
