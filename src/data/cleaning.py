"""Auditable STEP 02 transaction cleaning and quality-flag transformations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import DataValidationError


@dataclass(frozen=True)
class OutlierThresholds:
    """Training-population IQR and fixed domain thresholds."""

    quantity_lower: float
    quantity_upper: float
    price_lower: float
    price_upper: float
    quantity_absolute_domain_limit: float
    price_domain_limit: float


@dataclass(frozen=True)
class CleaningSummary:
    """Counts describing the deterministic cleaning result."""

    input_rows: int
    exact_duplicates_removed: int
    output_rows: int
    missing_customer_rows: int
    return_rows: int
    cancelled_invoice_rows: int
    non_positive_price_rows: int
    administrative_rows: int
    statistical_outlier_rows: int
    likely_data_error_rows: int


def normalize_description(series: pd.Series) -> pd.Series:
    """Normalize description case and whitespace without inventing missing text."""
    normalized = series.astype("string").str.upper()
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def _administrative_mask(stock_codes: pd.Series, config: dict[str, Any]) -> pd.Series:
    exact = {str(value).strip().upper() for value in config["administrative_stock_codes"]}
    patterns = [
        re.compile(str(value), flags=re.IGNORECASE)
        for value in config["administrative_stock_code_patterns"]
    ]
    codes = stock_codes.astype("string").str.upper().str.strip()
    mask = codes.isin(exact)
    for pattern in patterns:
        mask |= codes.str.match(pattern, na=False)
    return mask.astype(bool)


def _iqr_bounds(values: pd.Series, multiplier: float) -> tuple[float, float]:
    eligible = pd.to_numeric(values, errors="coerce").dropna().astype("float64")
    if eligible.empty:
        raise DataValidationError("Cannot fit outlier thresholds on an empty training population")
    first, third = eligible.quantile([0.25, 0.75]).tolist()
    spread = third - first
    return float(first - multiplier * spread), float(third + multiplier * spread)


def fit_outlier_thresholds(
    transactions: pd.DataFrame,
    *,
    training_customer_ids: set[str],
    config: dict[str, Any],
) -> OutlierThresholds:
    """Fit IQR thresholds using identified training customers only."""
    train = transactions[transactions["customer_id"].isin(training_customer_ids)]
    multiplier = float(config["iqr_multiplier"])
    quantity_lower, quantity_upper = _iqr_bounds(train["quantity"], multiplier)
    positive_prices = train.loc[train["price"] > 0, "price"]
    price_lower, price_upper = _iqr_bounds(positive_prices, multiplier)
    return OutlierThresholds(
        quantity_lower=quantity_lower,
        quantity_upper=quantity_upper,
        price_lower=price_lower,
        price_upper=price_upper,
        quantity_absolute_domain_limit=float(config["quantity_absolute_domain_limit"]),
        price_domain_limit=float(config["price_domain_limit"]),
    )


def clean_transactions(
    transactions: pd.DataFrame,
    *,
    cleaning_config: dict[str, Any],
    outlier_thresholds: OutlierThresholds | None = None,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Remove exact duplicates and add deterministic audit and eligibility flags."""
    input_rows = len(transactions)
    exact_duplicate = transactions.duplicated(keep="first")
    cleaned = transactions.loc[~exact_duplicate].copy()
    cleaned["description_normalized"] = normalize_description(cleaned["description"])
    cleaned["is_missing_customer_id"] = cleaned["customer_id"].isna()
    cleaned["is_cancelled_invoice"] = (
        cleaned["invoice"].astype("string").str.upper().str.startswith("C", na=False)
    )
    cleaned["is_return"] = cleaned["quantity"].lt(0)
    cleaned["is_non_positive_price"] = cleaned["price"].le(0)
    cleaned["is_zero_quantity"] = cleaned["quantity"].eq(0)
    cleaned["is_administrative_line"] = _administrative_mask(cleaned["stock_code"], cleaning_config)
    cleaned["is_product"] = ~cleaned["is_administrative_line"]

    if outlier_thresholds is None:
        cleaned["is_quantity_iqr_outlier"] = False
        cleaned["is_price_iqr_outlier"] = False
        quantity_limit = float(cleaning_config["outliers"]["quantity_absolute_domain_limit"])
        price_limit = float(cleaning_config["outliers"]["price_domain_limit"])
    else:
        cleaned["is_quantity_iqr_outlier"] = ~cleaned["quantity"].between(
            outlier_thresholds.quantity_lower,
            outlier_thresholds.quantity_upper,
            inclusive="both",
        )
        cleaned["is_price_iqr_outlier"] = ~cleaned["price"].between(
            outlier_thresholds.price_lower,
            outlier_thresholds.price_upper,
            inclusive="both",
        )
        quantity_limit = outlier_thresholds.quantity_absolute_domain_limit
        price_limit = outlier_thresholds.price_domain_limit

    cleaned["is_statistical_outlier"] = (
        cleaned["is_quantity_iqr_outlier"] | cleaned["is_price_iqr_outlier"]
    )
    cleaned["is_likely_data_error"] = cleaned["quantity"].abs().gt(quantity_limit) | cleaned[
        "price"
    ].gt(price_limit)
    cleaned["is_valid_merchandise"] = (
        cleaned["is_product"]
        & ~cleaned["is_non_positive_price"]
        & ~cleaned["is_zero_quantity"]
        & ~cleaned["is_likely_data_error"]
    )
    cleaned["is_positive_purchase"] = (
        cleaned["is_valid_merchandise"]
        & cleaned["quantity"].gt(0)
        & ~cleaned["is_cancelled_invoice"]
    )
    signed = cleaned["quantity"].astype("float64") * cleaned["price"]
    cleaned["signed_merchandise_value"] = np.where(cleaned["is_valid_merchandise"], signed, 0.0)
    cleaned["gross_positive_value"] = np.where(cleaned["is_positive_purchase"], signed, 0.0)
    cleaned = cleaned.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)

    validate_clean_transactions(cleaned)
    summary = CleaningSummary(
        input_rows=input_rows,
        exact_duplicates_removed=int(exact_duplicate.sum()),
        output_rows=len(cleaned),
        missing_customer_rows=int(cleaned["is_missing_customer_id"].sum()),
        return_rows=int(cleaned["is_return"].sum()),
        cancelled_invoice_rows=int(cleaned["is_cancelled_invoice"].sum()),
        non_positive_price_rows=int(cleaned["is_non_positive_price"].sum()),
        administrative_rows=int(cleaned["is_administrative_line"].sum()),
        statistical_outlier_rows=int(cleaned["is_statistical_outlier"].sum()),
        likely_data_error_rows=int(cleaned["is_likely_data_error"].sum()),
    )
    return cleaned, summary


def validate_clean_transactions(frame: pd.DataFrame) -> None:
    """Validate invariants of the auditable cleaned transaction table."""
    required = {
        "description_normalized",
        "is_missing_customer_id",
        "is_cancelled_invoice",
        "is_return",
        "is_non_positive_price",
        "is_administrative_line",
        "is_product",
        "is_statistical_outlier",
        "is_likely_data_error",
        "is_valid_merchandise",
        "is_positive_purchase",
        "signed_merchandise_value",
        "gross_positive_value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataValidationError(f"Clean transaction columns missing: {missing}")
    if bool(
        frame.duplicated(
            subset=[
                "invoice",
                "stock_code",
                "description",
                "quantity",
                "invoice_date",
                "price",
                "customer_id",
                "country",
            ]
        ).any()
    ):
        raise DataValidationError("Exact duplicate transaction lines remain after cleaning")
    if not frame["invoice_date"].is_monotonic_increasing:
        raise DataValidationError("Clean transactions are not chronologically ordered")
    if bool((frame["is_positive_purchase"] & frame["is_return"]).any()):
        raise DataValidationError("Return rows cannot be positive purchase events")
