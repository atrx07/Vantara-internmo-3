"""Development-only rolling-snapshot remediation for the versioned CLV v2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.data.splits import validate_customer_split
from src.data.validation import DataValidationError
from src.models.clv import TrainingTargetRangeRegressor
from src.models.common import configure_mlflow, log_mlflow_run, regression_metrics
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)

MODEL_VERSION = "vantara-clv-production-v2"
TARGET_NAME = "clv_180d_target"
ORIGINAL_V1_HISTORY_HASHES = {
    "reports/model_freeze/model_freeze.json": (
        "be64ed61ab5d4dc5f0a849a4101a4d5aa986ecf8d3bbbc24cce5e91e5562ba84"
    ),
    "models_artifacts/clv/production_clv.joblib": (
        "b306ef3e2435baafbec93b1d113faeff1e6cd8a3d5eeea9f1f14ea00b97dd6eb"
    ),
}
ORIGINAL_FINAL_EVIDENCE_HASHES = {
    "autoencoder_test_scores.csv": (
        "e989e823d1a099dee4783c4596381e6ddf1c7bbf06c540ba7cc1d07e86a30f7f"
    ),
    "churn_test_predictions.csv": (
        "0867f8c0be0fe30ef89e90de358f62ee6071be67c09fc68d4ddaf9448196d5ba"
    ),
    "clv_test_predictions.csv": "217c4d4f98d5c089bb84b15595523941ed3ec338762cd1470b9d80bd42f54cce",
    "final_evaluation_execution_lock.json": (
        "8b058205ba593c52587b57e6575f0303eaeaf60b22e89c7c68ce2e4273f91ca3"
    ),
    "final_metrics.json": "872cc6f549db63a8d9728aaa8bd4796b67507569bb5c5e846bbdfa4458df3a46",
    "lstm_test_predictions.csv": "69c360551853c51cc9b5e2aaa9e20df415091774a06b41bc38e2382863ae937b",
    "next_category_test_predictions.csv": (
        "19a1d92a1ab7352f4202344419db6135e99fd2b11cea2b24df49b343b305dad6"
    ),
}

CLV_V2_FEATURES = (
    "spend_30d",
    "spend_60d",
    "spend_90d",
    "spend_180d",
    "orders_30d",
    "orders_60d",
    "orders_90d",
    "orders_180d",
    "recent_spend_velocity",
    "recent_order_frequency_velocity",
    "avg_order_value_30d",
    "avg_order_value_60d",
    "avg_order_value_90d",
    "avg_order_value_180d",
    "maximum_recent_order_value",
    "recent_order_value_std",
    "active_month_count",
    "customer_tenure_days",
    "recency_days",
    "purchase_gap_mean_days",
    "purchase_gap_std_days",
    "return_value_ratio",
    "bulk_order_share",
    "high_value_order_share",
    "lifetime_orders",
    "lifetime_positive_spend",
    "lifetime_net_revenue",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_original_final_evidence(project_root: Path) -> dict[str, str]:
    """Fail closed unless every historical STEP 06 final file matches its pre-remediation hash."""
    evidence = project_root / "reports" / "final_evaluation"
    actual: dict[str, str] = {}
    for name, expected in ORIGINAL_FINAL_EVIDENCE_HASHES.items():
        path = evidence / name
        if not path.is_file():
            raise DataValidationError(f"Missing immutable STEP 06 evidence: {path}")
        actual[name] = _sha256(path)
        if actual[name] != expected:
            raise DataValidationError(f"Immutable STEP 06 evidence changed: {path}")
    return actual


def verify_original_v1_history(project_root: Path) -> dict[str, str]:
    """Fail closed unless the historical v1 freeze and CLV artifact remain unchanged."""
    actual: dict[str, str] = {}
    for relative, expected in ORIGINAL_V1_HISTORY_HASHES.items():
        path = project_root / relative
        if not path.is_file():
            raise DataValidationError(f"Missing historical v1 file: {path}")
        actual[relative] = _sha256(path)
        if actual[relative] != expected:
            raise DataValidationError(f"Historical v1 file changed: {path}")
    return actual


class HurdleCLVRegressor(RegressorMixin, BaseEstimator):
    """Expected-value hurdle model: purchase probability times positive-spend magnitude."""

    def __init__(self, classifier: BaseEstimator, magnitude_model: BaseEstimator) -> None:
        self.classifier = classifier
        self.magnitude_model = magnitude_model

    def fit(
        self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray
    ) -> HurdleCLVRegressor:
        """Fit zero/positive occurrence and positive-only magnitude components."""
        values = np.asarray(target, dtype="float64")
        if np.any(values < 0):
            raise ValueError("CLV hurdle targets must be nonnegative")
        positive = values > 0
        self.target_max_ = float(values.max(initial=0.0))
        self.constant_probability_ = float(positive.mean())
        self.classifier_ = None
        self.magnitude_model_ = None
        if positive.any() and not positive.all():
            self.classifier_ = clone(self.classifier).fit(features, positive.astype("int8"))
        if positive.any():
            self.magnitude_model_ = clone(self.magnitude_model).fit(
                np.asarray(features)[positive], values[positive]
            )
        return self

    def positive_probability(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the fitted probability of positive forward spend."""
        if self.classifier_ is None:
            return np.full(len(features), self.constant_probability_, dtype="float64")
        return np.asarray(self.classifier_.predict_proba(features)[:, 1], dtype="float64")

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return nonnegative expected CLV, safely handling all-zero training targets."""
        probability = self.positive_probability(features)
        if self.magnitude_model_ is None:
            return np.zeros(len(features), dtype="float64")
        magnitude = np.asarray(self.magnitude_model_.predict(features), dtype="float64")
        return np.clip(probability * np.maximum(magnitude, 0.0), 0.0, self.target_max_)


def _order_table(transactions: pd.DataFrame) -> pd.DataFrame:
    positive = transactions.loc[transactions["is_positive_purchase"].fillna(False)].copy()
    return (
        positive.groupby(["customer_id", "invoice"], observed=True)
        .agg(
            invoice_date=("invoice_date", "min"),
            order_value=("gross_positive_value", "sum"),
            order_units=("quantity", "sum"),
        )
        .reset_index()
        .sort_values(["customer_id", "invoice_date", "invoice"], kind="mergesort")
    )


def _window_orders(orders: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
    return orders.loc[
        orders["invoice_date"].ge(cutoff - pd.Timedelta(days=days))
        & orders["invoice_date"].lt(cutoff)
    ]


def build_clv_features_for_cutoff(
    transactions: pd.DataFrame,
    *,
    customer_ids: Sequence[str],
    cutoff: pd.Timestamp,
    bulk_order_units: int,
    high_value_order_gbp: float,
) -> pd.DataFrame:
    """Build CLV v2 features using only transactions strictly before one cutoff."""
    identifiers = [str(value) for value in customer_ids]
    identifier_set = set(identifiers)
    history = transactions.loc[
        transactions["customer_id"].astype("string").isin(identifier_set)
        & transactions["invoice_date"].lt(cutoff)
    ].copy()
    if history.empty:
        raise DataValidationError("CLV v2 snapshot history is empty")
    orders = _order_table(history)
    base = pd.DataFrame(index=pd.Index(identifiers, name="customer_id"))
    grouped = orders.groupby("customer_id", observed=True)
    base["lifetime_orders"] = grouped["invoice"].nunique()
    base["lifetime_positive_spend"] = grouped["order_value"].sum()
    base["customer_tenure_days"] = (
        (cutoff - grouped["invoice_date"].min()).dt.total_seconds().div(86400.0)
    )
    base["recency_days"] = (cutoff - grouped["invoice_date"].max()).dt.total_seconds().div(86400.0)

    gaps = orders.copy()
    gaps["gap_days"] = (
        gaps.groupby("customer_id", observed=True)["invoice_date"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
    )
    gap_stats = gaps.groupby("customer_id", observed=True)["gap_days"].agg(["mean", "std"])
    base["purchase_gap_mean_days"] = gap_stats["mean"]
    base["purchase_gap_std_days"] = gap_stats["std"]

    for days in (30, 60, 90, 180):
        recent = _window_orders(orders, cutoff, days)
        recent_grouped = recent.groupby("customer_id", observed=True)
        base[f"spend_{days}d"] = recent_grouped["order_value"].sum()
        base[f"orders_{days}d"] = recent_grouped["invoice"].nunique()
        base[f"avg_order_value_{days}d"] = recent_grouped["order_value"].mean()

    recent_180 = _window_orders(orders, cutoff, 180)
    recent_grouped = recent_180.groupby("customer_id", observed=True)
    base["maximum_recent_order_value"] = recent_grouped["order_value"].max()
    base["recent_order_value_std"] = recent_grouped["order_value"].std(ddof=0)
    active = recent_180.assign(active_month=recent_180["invoice_date"].dt.to_period("M"))
    base["active_month_count"] = active.groupby("customer_id", observed=True)[
        "active_month"
    ].nunique()
    base["bulk_order_share"] = recent_grouped["order_units"].apply(
        lambda values: float(values.ge(bulk_order_units).mean())
    )
    base["high_value_order_share"] = recent_grouped["order_value"].apply(
        lambda values: float(values.ge(high_value_order_gbp).mean())
    )

    previous_30 = orders.loc[
        orders["invoice_date"].ge(cutoff - pd.Timedelta(days=60))
        & orders["invoice_date"].lt(cutoff - pd.Timedelta(days=30))
    ]
    previous_grouped = previous_30.groupby("customer_id", observed=True)
    prior_spend = previous_grouped["order_value"].sum().reindex(base.index, fill_value=0.0)
    prior_orders = previous_grouped["invoice"].nunique().reindex(base.index, fill_value=0.0)
    base["recent_spend_velocity"] = (base["spend_30d"].fillna(0.0) - prior_spend) / 30.0
    base["recent_order_frequency_velocity"] = (base["orders_30d"].fillna(0.0) - prior_orders) / 30.0

    valid = history.loc[history["is_valid_merchandise"].fillna(False)].copy()
    signed = valid.groupby("customer_id", observed=True)["signed_merchandise_value"].sum()
    base["lifetime_net_revenue"] = signed
    valid_180 = valid.loc[valid["invoice_date"].ge(cutoff - pd.Timedelta(days=180))]
    positive_value = (
        valid_180.loc[valid_180["signed_merchandise_value"].gt(0)]
        .groupby("customer_id", observed=True)["signed_merchandise_value"]
        .sum()
    )
    return_value = (
        valid_180.loc[valid_180["signed_merchandise_value"].lt(0)]
        .assign(return_value=lambda frame: frame["signed_merchandise_value"].abs())
        .groupby("customer_id", observed=True)["return_value"]
        .sum()
    )
    base["return_value_ratio"] = (
        return_value.reindex(base.index, fill_value=0.0)
        / positive_value.reindex(base.index).replace(0.0, np.nan)
    ).fillna(0.0)
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base = base.reset_index()
    return base.loc[:, ["customer_id", *CLV_V2_FEATURES]]


def _rolling_cutoffs(
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
    *,
    initial_history_days: int,
    horizon_days: int,
    frequency: str,
) -> pd.DatetimeIndex:
    earliest = observation_start.normalize() + pd.Timedelta(days=initial_history_days)
    earliest = pd.offsets.MonthBegin().rollforward(earliest)
    latest = (observation_end - pd.Timedelta(days=horizon_days)).to_period("M").start_time
    cutoffs = pd.date_range(earliest, latest, freq=frequency)
    if cutoffs.empty:
        raise DataValidationError("No complete rolling CLV horizon is available")
    return cutoffs


def build_rolling_clv_dataset(
    transactions: pd.DataFrame,
    split: pd.DataFrame,
    *,
    horizon_days: int,
    cutoff_frequency: str,
    initial_history_days: int,
    minimum_customer_history_days: int,
    minimum_history_orders: int,
    bulk_order_units: int,
    high_value_order_gbp: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build development-only snapshots while preserving the original customer partition."""
    validate_customer_split(split)
    partition = split["partition"].astype("string")
    development_split = split.loc[partition.isin(["train", "validation"])].copy()
    development_ids = set(development_split["customer_id"].astype(str))
    original_test_ids = set(split.loc[partition.eq("test"), "customer_id"].astype(str))
    filtered = transactions.loc[
        transactions["customer_id"].astype("string").isin(development_ids)
    ].copy()
    observed_ids = set(filtered["customer_id"].dropna().astype(str))
    if observed_ids & original_test_ids:
        raise DataValidationError("Original test customer reached CLV v2 development rows")
    observation_start = pd.Timestamp(filtered["invoice_date"].min())
    observation_end = pd.Timestamp(filtered["invoice_date"].max())
    cutoffs = _rolling_cutoffs(
        observation_start,
        observation_end,
        initial_history_days=initial_history_days,
        horizon_days=horizon_days,
        frequency=cutoff_frequency,
    )
    orders = _order_table(filtered)
    partition_lookup = development_split.set_index("customer_id")["partition"].astype("string")
    snapshots: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        history_orders = orders.loc[orders["invoice_date"].lt(cutoff)]
        history = history_orders.groupby("customer_id", observed=True).agg(
            history_orders=("invoice", "nunique"),
            first_purchase=("invoice_date", "min"),
        )
        history["history_days"] = (
            (cutoff - history["first_purchase"]).dt.total_seconds().div(86400.0)
        )
        eligible = (
            history.loc[
                history["history_orders"].ge(minimum_history_orders)
                & history["history_days"].ge(minimum_customer_history_days)
            ]
            .index.astype(str)
            .tolist()
        )
        if not eligible:
            continue
        features = build_clv_features_for_cutoff(
            filtered,
            customer_ids=eligible,
            cutoff=cutoff,
            bulk_order_units=bulk_order_units,
            high_value_order_gbp=high_value_order_gbp,
        )
        target_end = cutoff + pd.Timedelta(days=horizon_days)
        future = filtered.loc[
            filtered["customer_id"].astype("string").isin(eligible)
            & filtered["invoice_date"].ge(cutoff)
            & filtered["invoice_date"].lt(target_end)
            & filtered["is_valid_merchandise"].fillna(False),
            ["customer_id", "signed_merchandise_value"],
        ]
        targets = future.groupby("customer_id", observed=True)["signed_merchandise_value"].sum()
        feature_max = (
            filtered.loc[
                filtered["customer_id"].astype("string").isin(eligible)
                & filtered["invoice_date"].lt(cutoff)
            ]
            .groupby("customer_id", observed=True)["invoice_date"]
            .max()
        )
        features[TARGET_NAME] = features["customer_id"].map(targets).fillna(0.0).clip(lower=0.0)
        features["cutoff_timestamp"] = cutoff
        features["feature_max_timestamp"] = features["customer_id"].map(feature_max)
        features["target_window_end"] = target_end
        features["observation_end"] = observation_end
        features["partition"] = features["customer_id"].map(partition_lookup)
        snapshots.append(features)
    if not snapshots:
        raise DataValidationError("No eligible CLV v2 rolling snapshots were generated")
    dataset = pd.concat(snapshots, ignore_index=True)
    dataset = dataset.sort_values(
        ["cutoff_timestamp", "customer_id"], kind="mergesort", ignore_index=True
    )
    if set(dataset["customer_id"].astype(str)) & original_test_ids:
        raise DataValidationError("Original test customer appears in CLV v2 dataset")
    if not dataset["feature_max_timestamp"].lt(dataset["cutoff_timestamp"]).all():
        raise DataValidationError("CLV v2 feature row crossed its cutoff")
    if not dataset["target_window_end"].le(dataset["observation_end"]).all():
        raise DataValidationError("CLV v2 target row lacks a complete 180-day horizon")
    values = dataset[TARGET_NAME].astype("float64")
    summary = {
        "model_version": MODEL_VERSION,
        "snapshot_count": len(dataset),
        "development_customer_count": dataset["customer_id"].nunique(),
        "train_customer_count": dataset.loc[
            dataset["partition"].eq("train"), "customer_id"
        ].nunique(),
        "validation_customer_count": dataset.loc[
            dataset["partition"].eq("validation"), "customer_id"
        ].nunique(),
        "original_test_customer_count": len(original_test_ids),
        "original_test_customer_intersection": 0,
        "snapshot_date_min": dataset["cutoff_timestamp"].min().isoformat(),
        "snapshot_date_max": dataset["cutoff_timestamp"].max().isoformat(),
        "horizon_days": horizon_days,
        "target_zero_rate": float(values.eq(0.0).mean()),
        "target_mean": float(values.mean()),
        "target_median": float(values.median()),
        "target_p75": float(values.quantile(0.75)),
        "target_p90": float(values.quantile(0.90)),
        "target_p95": float(values.quantile(0.95)),
        "target_p99": float(values.quantile(0.99)),
        "target_max": float(values.max()),
        "feature_count": len(CLV_V2_FEATURES),
        "feature_names": list(CLV_V2_FEATURES),
        "all_features_strictly_before_cutoff": True,
        "all_targets_have_complete_horizon": True,
        "original_held_out_test_reused": False,
    }
    return dataset, summary


def grouped_clv_folds(
    table: pd.DataFrame, *, n_splits: int, stratification_bins: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified customer-group folds and prove customer isolation."""
    target = table[TARGET_NAME].astype("float64")
    bins = pd.qcut(
        np.log1p(target).rank(method="first"),
        q=min(stratification_bins, len(target)),
        labels=False,
        duplicates="drop",
    )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = table["customer_id"].astype(str).to_numpy()
    folds = list(splitter.split(table.loc[:, CLV_V2_FEATURES], bins, groups=groups))
    for train_index, validation_index in folds:
        if set(groups[train_index]) & set(groups[validation_index]):
            raise DataValidationError("Customer crossed a CLV v2 grouped fold")
    return folds


def _xgb_parameters(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {
        **dict(config["xgboost"]),
        "tree_method": "hist",
        "n_jobs": 2,
        "random_state": seed,
    }


def candidate_estimators(config: Mapping[str, Any], *, seed: int) -> dict[str, BaseEstimator]:
    """Return all required CLV v2 candidates with fold-local preprocessing."""
    parameters = _xgb_parameters(config, seed)
    imputed_log_xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(objective="reg:squarederror", **parameters),
            ),
        ]
    )
    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    tweedie_parameters = {
        **parameters,
        "objective": "reg:tweedie",
        "tweedie_variance_power": float(config["tweedie_variance_power"]),
    }
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **parameters,
    )
    magnitude = XGBRegressor(**tweedie_parameters)
    return {
        "ridge_log1p_baseline": TrainingTargetRangeRegressor(
            TransformedTargetRegressor(
                regressor=ridge,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=True,
            )
        ),
        "xgboost_log1p": TrainingTargetRangeRegressor(
            TransformedTargetRegressor(
                regressor=imputed_log_xgb,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=True,
            )
        ),
        "xgboost_squared_error": TrainingTargetRangeRegressor(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(objective="reg:squarederror", **parameters)),
                ]
            )
        ),
        "xgboost_tweedie": TrainingTargetRangeRegressor(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(**tweedie_parameters)),
                ]
            )
        ),
        "xgboost_hurdle": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HurdleCLVRegressor(
                        classifier=classifier,
                        magnitude_model=magnitude,
                    ),
                ),
            ]
        ),
    }


def evaluate_candidates(
    dataset: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    seed: int,
    mlflow_tags: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BaseEstimator]]:
    """Evaluate candidates with grouped CV and untouched original validation customers."""
    train = dataset.loc[dataset["partition"].eq("train")].reset_index(drop=True)
    validation = dataset.loc[dataset["partition"].eq("validation")].reset_index(drop=True)
    if train.empty or validation.empty:
        raise DataValidationError("CLV v2 requires both original train and validation customers")
    folds = grouped_clv_folds(
        train,
        n_splits=int(config["cv_folds"]),
        stratification_bins=int(config["stratification_bins"]),
        seed=seed,
    )
    estimators = candidate_estimators(config, seed=seed)
    fold_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    for name, estimator in estimators.items():
        started = time.perf_counter()
        model_fold_records: list[dict[str, Any]] = []
        for fold_number, (fit_index, score_index) in enumerate(folds, start=1):
            fold_estimator = clone(estimator)
            fold_estimator.fit(
                train.loc[fit_index, CLV_V2_FEATURES], train.loc[fit_index, TARGET_NAME]
            )
            predictions = np.maximum(
                fold_estimator.predict(train.loc[score_index, CLV_V2_FEATURES]), 0.0
            )
            metrics = regression_metrics(train.loc[score_index, TARGET_NAME], predictions)
            record = {
                "model": name,
                "fold": fold_number,
                "train_rows": len(fit_index),
                "validation_rows": len(score_index),
                "train_customers": train.loc[fit_index, "customer_id"].nunique(),
                "validation_customers": train.loc[score_index, "customer_id"].nunique(),
                **metrics,
            }
            model_fold_records.append(record)
            fold_records.append(record)
        fitted = clone(estimator).fit(train.loc[:, CLV_V2_FEATURES], train[TARGET_NAME])
        validation_predictions = np.maximum(fitted.predict(validation.loc[:, CLV_V2_FEATURES]), 0.0)
        validation_metrics = regression_metrics(validation[TARGET_NAME], validation_predictions)
        elapsed = time.perf_counter() - started
        r2_values = np.asarray([record["r2"] for record in model_fold_records], dtype="float64")
        aggregate = {
            "cv_r2_mean": float(r2_values.mean()),
            "cv_r2_median": float(np.median(r2_values)),
            "cv_r2_min": float(r2_values.min()),
            "cv_r2_std": float(r2_values.std(ddof=0)),
            "cv_mae_mean": float(np.mean([record["mae"] for record in model_fold_records])),
            "cv_rmse_mean": float(np.mean([record["rmse"] for record in model_fold_records])),
        }
        eligibility_failures: list[str] = []
        if aggregate["cv_r2_mean"] <= 0:
            eligibility_failures.append("mean CV R2 is not positive")
        if aggregate["cv_r2_median"] <= 0:
            eligibility_failures.append("median CV R2 is not positive")
        if validation_metrics["r2"] <= 0:
            eligibility_failures.append("validation R2 is not positive")
        if aggregate["cv_r2_min"] < float(config["catastrophic_fold_r2"]):
            eligibility_failures.append("minimum fold R2 is catastrophically unstable")
        eligible = not eligibility_failures
        flat_metrics = {
            **aggregate,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        run_id = log_mlflow_run(
            run_name=f"clv_v2_{name}",
            family="clv_v2_remediation",
            parameters={"candidate": name, **dict(config["xgboost"])},
            metrics=flat_metrics,
            tags={**dict(mlflow_tags), "original_held_out_test_reused": "false"},
            training_seconds=elapsed,
        )
        comparison_records.append(
            {
                "model": name,
                **aggregate,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "eligible": eligible,
                "eligibility_reason": "eligible" if eligible else "; ".join(eligibility_failures),
                "training_seconds": elapsed,
                "mlflow_run_id": run_id,
                "original_held_out_test_reused": False,
            }
        )
    comparison = pd.DataFrame(comparison_records).sort_values(
        [
            "eligible",
            "cv_r2_mean",
            "cv_r2_min",
            "cv_r2_std",
            "validation_r2",
            "validation_rmse",
        ],
        ascending=[False, False, False, True, False, True],
        ignore_index=True,
    )
    return comparison, pd.DataFrame(fold_records), estimators


def select_production_candidate(comparison: pd.DataFrame) -> pd.Series:
    """Select only a stable, positive-generalization candidate under the locked v2 rule."""
    eligible = comparison.loc[comparison["eligible"]].copy()
    if eligible.empty:
        raise DataValidationError(
            "No CLV v2 candidate passed grouped-CV and validation eligibility; freeze refused"
        )
    return eligible.iloc[0]


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")


def run_clv_remediation(
    config_path: str | Path = "config/config.yaml", *, freeze: bool = False
) -> dict[str, Any]:
    """Build, evaluate, and optionally freeze CLV v2 without touching original test evidence."""
    config_file = Path(config_path).resolve()
    root = config_file.parents[1]
    config = load_config(config_file)
    configure_logging(str(config["logging"]["level"]))
    remediation = config["clv_remediation"]
    if remediation["model_version"] != MODEL_VERSION:
        raise DataValidationError("CLV remediation model version does not match the v2 contract")
    original_before = verify_original_final_evidence(root)
    v1_history_before = verify_original_v1_history(root)
    split = pd.read_parquet(
        resolve_project_path(config["outputs"]["customer_split"], project_root=root)
    )
    transactions = pd.read_parquet(
        resolve_project_path(config["cleaning"]["interim_transactions"], project_root=root)
    )
    dataset, summary = build_rolling_clv_dataset(
        transactions,
        split,
        horizon_days=int(remediation["horizon_days"]),
        cutoff_frequency=str(remediation["cutoff_frequency"]),
        initial_history_days=int(remediation["initial_history_days"]),
        minimum_customer_history_days=int(remediation["minimum_customer_history_days"]),
        minimum_history_orders=int(remediation["minimum_history_orders"]),
        bulk_order_units=int(remediation["bulk_order_units"]),
        high_value_order_gbp=float(remediation["high_value_order_gbp"]),
    )
    dataset_path = resolve_project_path(remediation["dataset_path"], project_root=root)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path, index=False)
    evidence = resolve_project_path(remediation["evidence_directory"], project_root=root)
    evidence.mkdir(parents=True, exist_ok=True)
    _json_dump(evidence / "dataset_summary.json", summary)
    _json_dump(
        evidence / "original_step06_evidence_integrity.json",
        {
            "status": "PASS",
            "original_held_out_test_reused": False,
            "files": original_before,
            "v1_history": v1_history_before,
        },
    )
    configure_mlflow(
        resolve_project_path(config["modeling"]["tracking_directory"], project_root=root),
        "vantara-clv-v2-remediation",
    )
    tags = {
        "model_version": MODEL_VERSION,
        "split_version": str(config["features"]["split"]["version"]),
        "source_sha256": str(config["data"]["expected_sha256"]),
    }
    comparison, fold_metrics, estimators = evaluate_candidates(
        dataset,
        config=remediation,
        seed=int(config["project"]["random_seed"]),
        mlflow_tags=tags,
    )
    comparison.to_csv(evidence / "candidate_comparison.csv", index=False, lineterminator="\n")
    fold_metrics.to_csv(evidence / "fold_metrics.csv", index=False, lineterminator="\n")
    selected = select_production_candidate(comparison)
    result: dict[str, Any] = {
        "dataset_summary": summary,
        "selected_model": str(selected["model"]),
        "selected_metrics": {
            key: float(selected[key])
            for key in (
                "cv_r2_mean",
                "cv_r2_median",
                "cv_r2_min",
                "cv_r2_std",
                "validation_r2",
                "validation_mae",
                "validation_rmse",
            )
        },
        "freeze_requested": freeze,
    }
    if freeze:
        artifact_path = resolve_project_path(remediation["artifact_path"], project_root=root)
        freeze_path = evidence / "production_clv_v2_freeze.json"
        if artifact_path.exists() or freeze_path.exists():
            raise DataValidationError("CLV v2 production freeze already exists; overwrite refused")
        development = dataset.loc[dataset["partition"].isin(["train", "validation"])]
        final_estimator = clone(estimators[str(selected["model"])]).fit(
            development.loc[:, CLV_V2_FEATURES], development[TARGET_NAME]
        )
        metadata = {
            "model_family": "clv",
            "model_name": str(selected["model"]),
            "model_version": str(remediation["model_version"]),
            "supersedes": str(remediation["supersedes"]),
            "target": str(remediation["target"]),
            "selection_basis": str(remediation["selection_basis"]),
            "original_held_out_test_reused": False,
            "source_sha256": str(config["data"]["expected_sha256"]),
            "split_version": str(config["features"]["split"]["version"]),
            "feature_schema_version": "vantara-clv-features-v2",
            "feature_names": list(CLV_V2_FEATURES),
            "development_snapshot_count": len(development),
            "development_customer_count": development["customer_id"].nunique(),
            "metrics": result["selected_metrics"],
            "held_out_test_accessed": False,
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": final_estimator, "metadata": metadata}, artifact_path)
        reloaded = joblib.load(artifact_path)
        sample = development.loc[:, CLV_V2_FEATURES].head(3)
        reload_predictions = np.asarray(reloaded["pipeline"].predict(sample), dtype="float64")
        if not np.isfinite(reload_predictions).all() or (reload_predictions < 0).any():
            raise DataValidationError("Reloaded CLV v2 artifact failed inference validation")
        artifact_sha256 = _sha256(artifact_path)
        freeze_record = {
            **metadata,
            "artifact": str(Path(remediation["artifact_path"]).as_posix()),
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_path.stat().st_size,
            "selected_metrics": result["selected_metrics"],
            "candidate_comparison_sha256": _sha256(evidence / "candidate_comparison.csv"),
            "fold_metrics_sha256": _sha256(evidence / "fold_metrics.csv"),
            "dataset_summary_sha256": _sha256(evidence / "dataset_summary.json"),
            "original_step06_evidence_hashes": original_before,
            "original_v1_history_hashes": v1_history_before,
            "artifact_reload_validation": "PASS",
        }
        _json_dump(freeze_path, freeze_record)
        result["artifact"] = {
            "path": str(artifact_path),
            "sha256": artifact_sha256,
            "size_bytes": artifact_path.stat().st_size,
        }
    original_after = verify_original_final_evidence(root)
    v1_history_after = verify_original_v1_history(root)
    if original_after != original_before:
        raise DataValidationError("STEP 06 final evidence changed during CLV remediation")
    if v1_history_after != v1_history_before:
        raise DataValidationError("Historical CLV v1 freeze or artifact changed during remediation")
    LOGGER.info(
        "CLV v2 remediation evaluation complete",
        extra={"event": "clv_v2_remediation_complete", "selected_model": result["selected_model"]},
    )
    return result


def main() -> None:
    """Run the owner-authorized CLV v2 remediation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write the separate production v2 artifact after eligible evaluation",
    )
    arguments = parser.parse_args()
    result = run_clv_remediation(arguments.config, freeze=arguments.freeze)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
