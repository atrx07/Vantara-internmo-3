"""Canonical transaction CSV validation and server-owned batch feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from api.artifacts import ArtifactRegistry
from src.data.cleaning import clean_transactions
from src.data.validation import DataValidationError
from src.features.customer_features import build_customer_features
from src.features.product_artifacts import ProductArtifacts

CANONICAL_BATCH_COLUMNS = (
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
)


@dataclass(frozen=True)
class PreparedCustomer:
    """Server-owned feature and sequence inputs prepared from uploaded history."""

    customer_id: str
    country: str | None
    as_of: pd.Timestamp
    features: dict[str, float]
    sequence: dict[str, object] | None


def validate_batch_frame(frame: pd.DataFrame, *, maximum_rows: int) -> pd.DataFrame:
    """Validate and normalize the canonical transaction-level batch schema."""
    if tuple(frame.columns) != CANONICAL_BATCH_COLUMNS:
        raise DataValidationError(
            "Batch CSV columns must exactly match: " + ", ".join(CANONICAL_BATCH_COLUMNS)
        )
    if frame.empty:
        raise DataValidationError("Batch CSV must contain at least one transaction")
    if len(frame) > maximum_rows:
        raise DataValidationError(f"Batch CSV exceeds the {maximum_rows} row limit")
    normalized = frame.copy()
    for name in ("invoice", "stock_code", "description", "customer_id", "country"):
        normalized[name] = normalized[name].astype("string").str.strip()
    normalized["invoice_date"] = pd.to_datetime(normalized["invoice_date"], errors="coerce")
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce").astype("Int64")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    invalid = (
        normalized[["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id"]]
        .isna()
        .any(axis=1)
    )
    if invalid.any():
        raise DataValidationError(
            f"Batch CSV has {int(invalid.sum())} rows with invalid required values"
        )
    if not np.isfinite(normalized["price"].to_numpy(dtype="float64")).all():
        raise DataValidationError("Batch CSV price values must be finite")
    return normalized


def build_latest_sequence_payloads(
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    sequence_length: int,
    minimum_events: int,
) -> dict[str, dict[str, object]]:
    """Build the latest LSTM input per customer using only events before cutoff."""
    eligible = transactions.loc[
        transactions["customer_id"].notna()
        & transactions["invoice_date"].lt(cutoff)
        & transactions["is_positive_purchase"].fillna(False)
        & transactions["is_valid_merchandise"].fillna(False),
        [
            "customer_id",
            "invoice",
            "stock_code",
            "invoice_date",
            "quantity",
            "gross_positive_value",
        ],
    ].copy()
    if eligible.empty:
        return {}
    eligible = eligible.merge(
        taxonomy[["stock_code", "category_id"]], on="stock_code", how="left", validate="many_to_one"
    )
    eligible["category_index"] = eligible["category_id"].fillna(-1).astype("int64") + 1
    keys = ["customer_id", "invoice"]
    totals = (
        eligible.groupby(keys, observed=True)
        .agg(invoice_date=("invoice_date", "min"), order_amount=("gross_positive_value", "sum"))
        .reset_index()
    )
    category = (
        eligible.groupby([*keys, "category_index"], observed=True)
        .agg(category_value=("gross_positive_value", "sum"), category_quantity=("quantity", "sum"))
        .reset_index()
        .sort_values(
            [*keys, "category_value", "category_quantity", "category_index"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(keys, keep="first")
    )
    events = totals.merge(
        category[[*keys, "category_index"]], on=keys, validate="one_to_one"
    ).sort_values(["customer_id", "invoice_date", "invoice"], kind="mergesort")
    events["gap_days"] = (
        events.groupby("customer_id", observed=True)["invoice_date"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    payloads: dict[str, dict[str, object]] = {}
    for customer_id, group in events.groupby("customer_id", sort=True, observed=True):
        latest = group.tail(sequence_length)
        length = len(latest)
        if length < minimum_events:
            continue
        continuous = np.zeros((sequence_length, 2), dtype="float32")
        categories = np.zeros(sequence_length, dtype="int64")
        continuous[:length, 0] = np.log1p(
            latest["order_amount"].to_numpy(dtype="float64").clip(min=0.0)
        )
        continuous[:length, 1] = latest["gap_days"].to_numpy(dtype="float64")
        categories[:length] = latest["category_index"].to_numpy(dtype="int64")
        payloads[str(customer_id)] = {
            "continuous": continuous.tolist(),
            "categories": categories.tolist(),
            "length": length,
        }
    return payloads


def prepare_batch_customers(
    frame: pd.DataFrame,
    *,
    registry: ArtifactRegistry,
    config: dict[str, Any],
) -> list[PreparedCustomer]:
    """Clean uploaded history and derive frozen-schema features and LSTM sequences."""
    validated = validate_batch_frame(
        frame, maximum_rows=int(config["serving"]["batch_maximum_rows"])
    )
    cleaned, _ = clean_transactions(validated, cleaning_config=config["cleaning"])
    cutoff = pd.Timestamp(cleaned["invoice_date"].max()) + pd.Timedelta(seconds=1)
    customers = sorted(
        cleaned.loc[cleaned["is_positive_purchase"], "customer_id"].astype(str).unique()
    )
    if not customers:
        raise DataValidationError("Batch CSV has no customer with a valid positive purchase")
    taxonomy = registry.product_taxonomy
    preprocessing = registry.preprocessing_contracts
    unused = cast(Any, None)
    artifacts = ProductArtifacts(
        taxonomy_version="vantara-taxonomy-v1",
        fit_cutoff=cutoff,
        selected_clusters=30,
        taxonomy=taxonomy,
        reference_prices=registry.reference_prices,
        popularity=registry.product_popularity,
        taxonomy_candidates=pd.DataFrame(),
        top_terms={},
        vectorizer=unused,
        svd=unused,
        normalizer=unused,
        clusterer=unused,
    )
    features, _ = build_customer_features(
        cleaned,
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=set(customers),
        trend_window_days=int(config["snapshots"]["trend_window_days"]),
        markdown_price_ratio=float(config["features"]["markdown_price_ratio"]),
    )
    features["engagement_score"] = preprocessing["engagement"]["churn"].transform(features)
    sequences = build_latest_sequence_payloads(
        cleaned,
        taxonomy,
        cutoff=cutoff,
        sequence_length=int(config["deep_learning"]["lstm"]["sequence_length"]),
        minimum_events=int(config["deep_learning"]["lstm"]["minimum_history_events"]),
    )
    country_lookup = (
        cleaned.dropna(subset=["customer_id"])
        .sort_values("invoice_date", kind="mergesort")
        .drop_duplicates("customer_id", keep="last")
        .set_index("customer_id")["country"]
        .astype(str)
        .to_dict()
    )
    rows: list[PreparedCustomer] = []
    for _, row in features.iterrows():
        customer_id = str(row["customer_id"])
        payload = {name: float(row[name]) for name in registry.feature_names}
        rows.append(
            PreparedCustomer(
                customer_id=customer_id,
                country=country_lookup.get(customer_id),
                as_of=cutoff,
                features=payload,
                sequence=sequences.get(customer_id),
            )
        )
    return rows
