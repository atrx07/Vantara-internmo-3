"""Training-only correlation/VIF evidence and final churn feature-schema freeze."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import DataValidationError

NON_FEATURE_COLUMNS = {
    "customer_id",
    "cutoff_timestamp",
    "churn",
    "partition",
    "split_version",
}


@dataclass(frozen=True)
class FeatureSchemaEvidence:
    """Frozen churn schema plus train-only correlation and VIF evidence."""

    schema_version: str
    selected_features: tuple[str, ...]
    exclusions: dict[str, str]
    correlation_matrix: pd.DataFrame
    high_correlation_pairs: pd.DataFrame
    initial_vif: pd.DataFrame
    final_vif: pd.DataFrame


def _candidate_columns(features: pd.DataFrame) -> list[str]:
    numeric = set(features.select_dtypes(include=["number", "bool"]).columns)
    return [column for column in features.columns if column in numeric - NON_FEATURE_COLUMNS]


def calculate_vif(features: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF as the diagonal of the inverse standardized correlation matrix."""
    if features.empty or features.shape[1] < 2:
        return pd.DataFrame({"feature": list(features.columns), "vif": [1.0] * features.shape[1]})
    numeric = features.astype("float64")
    variance = numeric.var(ddof=0)
    if bool(variance.le(0).any()):
        constant = sorted(variance[variance.le(0)].index)
        raise DataValidationError(f"VIF input contains constant features: {constant}")
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    correlation = np.corrcoef(standardized.to_numpy(), rowvar=False)
    inverse = np.linalg.pinv(correlation, hermitian=True)
    values = np.maximum(np.diag(inverse), 1.0)
    return pd.DataFrame({"feature": numeric.columns, "vif": values}).sort_values(
        ["vif", "feature"], ascending=[False, True], ignore_index=True
    )


def _correlation_pairs(matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    columns = list(matrix.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            correlation = float(matrix.loc[left, right])
            if abs(correlation) >= threshold:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["feature_a", "feature_b", "correlation", "absolute_correlation"],
    ).sort_values(
        ["absolute_correlation", "feature_a", "feature_b"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def freeze_churn_feature_schema(
    features: pd.DataFrame,
    *,
    schema_version: str,
    correlation_threshold: float,
    vif_threshold: float,
) -> FeatureSchemaEvidence:
    """Select and freeze one non-redundant schema using training rows only."""
    if "partition" not in features or "churn" not in features:
        raise DataValidationError("Churn table must contain partition and churn columns")
    training = features.loc[features["partition"].astype("string").eq("train")]
    if training.empty:
        raise DataValidationError("No training rows are available for schema selection")
    candidates = _candidate_columns(training)
    if not candidates:
        raise DataValidationError("No numerical churn features are available for schema selection")
    matrix = training[candidates].corr(method="pearson")
    high_pairs = _correlation_pairs(matrix, correlation_threshold)
    selected = candidates.copy()
    exclusions: dict[str, str] = {}

    if "category_affinity_unknown" in selected:
        selected.remove("category_affinity_unknown")
        exclusions["category_affinity_unknown"] = (
            "Reference category removed from model matrix; full vector remains in feature table."
        )
    if "historical_customer_value" in selected and "net_spend" in selected:
        selected.remove("historical_customer_value")
        exclusions["historical_customer_value"] = (
            "Exact duplicate of net_spend in the current governed feature definition."
        )
    if (
        "gross_spend" in selected
        and "net_spend" in selected
        and abs(float(matrix.loc["gross_spend", "net_spend"])) >= correlation_threshold
    ):
        selected.remove("gross_spend")
        exclusions["gross_spend"] = (
            "Highly correlated with net_spend; net revenue is retained because it reflects returns."
        )

    non_constant = [column for column in selected if training[column].nunique(dropna=False) > 1]
    for column in sorted(set(selected) - set(non_constant)):
        exclusions[column] = "Zero variance in the training partition."
    selected = non_constant
    initial_vif = calculate_vif(training[selected])

    while len(selected) > 2:
        current_vif = calculate_vif(training[selected])
        worst = current_vif.iloc[0]
        if float(worst["vif"]) <= vif_threshold:
            break
        feature = str(worst["feature"])
        selected.remove(feature)
        exclusions[feature] = (
            f"Training VIF {float(worst['vif']):.6f} exceeded {vif_threshold:.2f}."
        )

    final_vif = calculate_vif(training[selected])
    if not selected:
        raise DataValidationError("Feature schema selection removed every candidate")
    if float(final_vif["vif"].max()) > vif_threshold + 1e-9:
        raise DataValidationError("Final churn feature schema still exceeds the VIF threshold")
    return FeatureSchemaEvidence(
        schema_version=schema_version,
        selected_features=tuple(selected),
        exclusions=exclusions,
        correlation_matrix=matrix,
        high_correlation_pairs=high_pairs,
        initial_vif=initial_vif,
        final_vif=final_vif,
    )


def write_feature_schema(evidence: FeatureSchemaEvidence, path: Path) -> None:
    """Persist canonical feature names/order/version and auditable exclusions."""
    payload: dict[str, Any] = {
        "schema_version": evidence.schema_version,
        "selected_features": list(evidence.selected_features),
        "feature_count": len(evidence.selected_features),
        "exclusions": evidence.exclusions,
        "selection_population": "training partition only",
        "correlation_method": "pearson",
        "vif_method": "diagonal of inverse standardized correlation matrix",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
