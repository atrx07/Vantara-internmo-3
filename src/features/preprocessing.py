"""Training-only model preprocessing contracts with explicit scaler separation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.splits import customer_ids_for_partition
from src.data.validation import DataValidationError


@dataclass(frozen=True)
class FittedPreprocessor:
    """One fitted feature-order and transformation contract."""

    name: str
    feature_names: tuple[str, ...]
    training_customer_ids: frozenset[str]
    pipeline: Pipeline
    scaled: bool

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform rows using the immutable fitted feature order."""
        missing = sorted(set(self.feature_names).difference(features.columns))
        if missing:
            raise DataValidationError(f"Preprocessing input missing features: {missing}")
        return np.asarray(
            self.pipeline.transform(features.loc[:, self.feature_names]), dtype="float64"
        )


@dataclass(frozen=True)
class PreprocessingContracts:
    """Separate scaled and unscaled preprocessing fitted on one training partition."""

    scaled: FittedPreprocessor
    unscaled: FittedPreprocessor


def fit_preprocessing_contracts(
    features: pd.DataFrame,
    split: pd.DataFrame,
    *,
    excluded_columns: set[str] | None = None,
) -> PreprocessingContracts:
    """Fit imputation/scaling only on training customers and freeze feature order."""
    excluded = {"customer_id", "cutoff_timestamp", "partition"}
    excluded.update(excluded_columns or set())
    feature_names = tuple(
        name
        for name in features.select_dtypes(include=["number", "bool"]).columns
        if name not in excluded
    )
    if not feature_names:
        raise DataValidationError("No numerical features are available for preprocessing")
    training_ids = customer_ids_for_partition(split, "train")
    training = features.loc[features["customer_id"].isin(training_ids), feature_names]
    if training.empty:
        raise DataValidationError("No training rows are available for preprocessing")

    scaled_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    ).fit(training)
    unscaled_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))]).fit(training)
    frozen_ids = frozenset(training_ids)
    return PreprocessingContracts(
        scaled=FittedPreprocessor(
            name="scaled_linear_distance_ann",
            feature_names=feature_names,
            training_customer_ids=frozen_ids,
            pipeline=scaled_pipeline,
            scaled=True,
        ),
        unscaled=FittedPreprocessor(
            name="unscaled_tree_boosting",
            feature_names=feature_names,
            training_customer_ids=frozen_ids,
            pipeline=unscaled_pipeline,
            scaled=False,
        ),
    )
