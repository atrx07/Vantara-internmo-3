"""Point-in-time customer feature construction for governed STEP 02 tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.validation import DataValidationError
from src.features.product_artifacts import ProductArtifacts

FEATURE_JUSTIFICATIONS: dict[str, str] = {
    "recency_days": "Recent purchasers are generally less likely to lapse.",
    "frequency_orders": "Repeated completed orders indicate established purchasing behavior.",
    "gross_spend": "Gross historical revenue measures customer commercial scale.",
    "net_spend": "Net spend reflects positive revenue after return behavior.",
    "avg_order_value": "Typical order value separates frequent small baskets from valuable orders.",
    "historical_customer_value": "Cumulative net revenue summarizes realized customer value.",
    "avg_basket_units": "Average purchased units captures basket depth.",
    "avg_distinct_products_per_order": "Product variety per order captures basket breadth.",
    "unique_product_count": "Lifetime product variety indicates breadth of engagement.",
    "customer_tenure_days": "Observed relationship length distinguishes new and mature customers.",
    "mean_interpurchase_gap_days": "Typical purchase cadence is predictive of the next purchase.",
    "variance_interpurchase_gap_days": "Cadence variability measures behavioral consistency.",
    "purchase_frequency_trend": "Recent weekly order slope captures acceleration or decline.",
    "seasonal_purchase_concentration": (
        "Quarter concentration captures seasonal purchasing patterns."
    ),
    "return_rate": "Returned units relative to purchased units capture post-purchase behavior.",
    "markdown_affinity_proxy": (
        "Share of eligible orders at least 10% below reference price proxies markdown affinity."
    ),
    "engagement_score": "Training-fitted RFM percentiles summarize engagement on a 0-100 scale.",
    "mean_product_popularity": (
        "Average training-only product popularity captures mainstream preference."
    ),
    "median_product_popularity": (
        "Median training-only product popularity reduces sensitivity to extreme items."
    ),
    "max_product_popularity": "Maximum popularity records exposure to highly common products.",
    "rare_product_share": "Share of rare training products captures niche purchasing behavior.",
}


@dataclass(frozen=True)
class EngagementPercentiles:
    """Frozen empirical RFM distributions fitted on training customers only."""

    recency: np.ndarray
    frequency: np.ndarray
    monetary: np.ndarray

    @classmethod
    def fit(cls, features: pd.DataFrame, training_customer_ids: set[str]) -> EngagementPercentiles:
        """Fit empirical distributions without observing validation/test customers."""
        training = features[features["customer_id"].isin(training_customer_ids)]
        if training.empty:
            raise DataValidationError(
                "Cannot fit engagement percentiles without training customers"
            )
        return cls(
            recency=np.sort(training["recency_days"].to_numpy(dtype="float64")),
            frequency=np.sort(training["frequency_orders"].to_numpy(dtype="float64")),
            monetary=np.sort(training["net_spend"].to_numpy(dtype="float64")),
        )

    @staticmethod
    def _percentile(values: pd.Series, fitted: np.ndarray) -> np.ndarray:
        return np.searchsorted(fitted, values.to_numpy(dtype="float64"), side="right") / len(fitted)

    def transform(self, features: pd.DataFrame) -> pd.Series:
        """Calculate the locked 40/30/30 engagement score using frozen distributions."""
        recency = self._percentile(features["recency_days"], self.recency)
        frequency = self._percentile(features["frequency_orders"], self.frequency)
        monetary = self._percentile(features["net_spend"], self.monetary)
        values = 100.0 * (0.40 * (1.0 - recency) + 0.30 * frequency + 0.30 * monetary)
        return pd.Series(np.clip(values, 0.0, 100.0), index=features.index, dtype="float64")


def _weekly_order_slope(
    positive_orders: pd.DataFrame,
    customer_ids: list[str],
    cutoff: pd.Timestamp,
    trend_window_days: int,
) -> pd.Series:
    start = cutoff - pd.Timedelta(days=trend_window_days)
    recent = positive_orders.loc[positive_orders["invoice_date"].ge(start)].copy()
    weeks = max(int(np.ceil(trend_window_days / 7)), 2)
    recent["week_index"] = (
        (recent["invoice_date"] - start).dt.total_seconds().floordiv(7 * 86400).astype(int)
    ).clip(0, weeks - 1)
    counts = recent.groupby(["customer_id", "week_index"], observed=True).size()
    x = np.arange(weeks, dtype="float64")
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    slopes: dict[str, float] = {}
    for customer_id in customer_ids:
        values = np.zeros(weeks, dtype="float64")
        if customer_id in counts.index.get_level_values(0):
            customer_counts = counts.loc[customer_id]
            values[customer_counts.index.to_numpy(dtype=int)] = customer_counts.to_numpy(
                dtype=float
            )
        slopes[customer_id] = float(np.dot(centered, values - values.mean()) / denominator)
    return pd.Series(slopes, name="purchase_frequency_trend", dtype="float64")


def _category_affinity(
    positive: pd.DataFrame,
    customer_ids: list[str],
    artifacts: ProductArtifacts,
) -> pd.DataFrame:
    categorized = positive.merge(
        artifacts.taxonomy[["stock_code", "category_id"]], on="stock_code", how="left"
    )
    categorized["category_id"] = categorized["category_id"].fillna(-1).astype(int)
    values = categorized.pivot_table(
        index="customer_id",
        columns="category_id",
        values="gross_positive_value",
        aggfunc="sum",
        fill_value=0.0,
        observed=True,
    )
    required_categories = [-1, *range(artifacts.selected_clusters)]
    values = values.reindex(columns=required_categories, fill_value=0.0)
    denominator = values.sum(axis=1).replace(0.0, np.nan)
    affinity = values.div(denominator, axis=0).fillna(0.0)
    affinity.columns = [
        "category_affinity_unknown" if value == -1 else f"category_affinity_{value:02d}"
        for value in required_categories
    ]
    return affinity.reindex(customer_ids, fill_value=0.0)


def build_customer_features(
    transactions: pd.DataFrame,
    *,
    customer_ids: list[str],
    cutoff: pd.Timestamp,
    artifacts: ProductArtifacts,
    training_customer_ids: set[str],
    trend_window_days: int,
    markdown_price_ratio: float,
) -> tuple[pd.DataFrame, EngagementPercentiles]:
    """Build all required features from rows strictly before the supplied cutoff."""
    history = transactions.loc[
        transactions["customer_id"].isin(customer_ids) & transactions["invoice_date"].lt(cutoff)
    ].copy()
    positive = history.loc[history["is_positive_purchase"]].copy()
    if positive.empty:
        raise DataValidationError("Snapshot history has no valid positive purchase events")

    base = pd.DataFrame({"customer_id": pd.Series(customer_ids, dtype="string")})
    positive_orders = (
        positive.groupby(["customer_id", "invoice"], observed=True)
        .agg(
            invoice_date=("invoice_date", "min"),
            order_value=("gross_positive_value", "sum"),
            basket_units=("quantity", "sum"),
            distinct_products=("stock_code", "nunique"),
        )
        .reset_index()
    )
    customer_orders = positive_orders.groupby("customer_id", observed=True).agg(
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min"),
        frequency_orders=("invoice", "nunique"),
        gross_spend=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        avg_basket_units=("basket_units", "mean"),
        avg_distinct_products_per_order=("distinct_products", "mean"),
    )
    base = base.join(customer_orders, on="customer_id")
    base["recency_days"] = (cutoff - base["last_purchase"]).dt.total_seconds() / 86400.0
    base["customer_tenure_days"] = (cutoff - base["first_purchase"]).dt.total_seconds() / 86400.0
    base["unique_product_count"] = base["customer_id"].map(
        positive.groupby("customer_id", observed=True)["stock_code"].nunique()
    )

    signed = (
        history.loc[history["is_valid_merchandise"]]
        .groupby("customer_id", observed=True)["signed_merchandise_value"]
        .sum()
    )
    base["net_spend"] = base["customer_id"].map(signed).fillna(0.0).clip(lower=0.0)
    base["historical_customer_value"] = base["net_spend"]

    order_dates = positive_orders.sort_values(
        ["customer_id", "invoice_date", "invoice"], kind="mergesort"
    )
    gaps = (
        order_dates.groupby("customer_id", observed=True)["invoice_date"].diff().dt.total_seconds()
    )
    order_dates["gap_days"] = gaps / 86400.0
    gap_stats = order_dates.groupby("customer_id", observed=True)["gap_days"].agg(
        mean_interpurchase_gap_days="mean",
        variance_interpurchase_gap_days=lambda values: (
            float(np.nanvar(values, ddof=0)) if values.notna().any() else 0.0
        ),
    )
    base = base.join(gap_stats, on="customer_id")
    base[["mean_interpurchase_gap_days", "variance_interpurchase_gap_days"]] = base[
        ["mean_interpurchase_gap_days", "variance_interpurchase_gap_days"]
    ].fillna(0.0)

    slopes = _weekly_order_slope(positive_orders, customer_ids, cutoff, trend_window_days)
    base["purchase_frequency_trend"] = base["customer_id"].map(slopes).fillna(0.0)

    positive_orders["year_quarter"] = positive_orders["invoice_date"].dt.to_period("Q").astype(str)
    quarter_counts = positive_orders.groupby(["customer_id", "year_quarter"], observed=True).size()
    seasonal = quarter_counts.groupby(level=0).max() / quarter_counts.groupby(level=0).sum()
    base["seasonal_purchase_concentration"] = base["customer_id"].map(seasonal).fillna(0.0)

    positive_units = positive.groupby("customer_id", observed=True)["quantity"].sum()
    returned_units = (
        history.loc[history["is_valid_merchandise"] & history["quantity"].lt(0)]
        .assign(returned_units=lambda frame: frame["quantity"].abs())
        .groupby("customer_id", observed=True)["returned_units"]
        .sum()
    )
    base["return_rate"] = (
        base["customer_id"].map(returned_units).fillna(0.0)
        / base["customer_id"].map(positive_units).replace(0, np.nan)
    ).fillna(0.0)

    eligible_markdown = positive.merge(
        artifacts.reference_prices[["stock_code", "reference_price"]], on="stock_code", how="inner"
    )
    eligible_markdown["markdown_like"] = eligible_markdown["price"].le(
        markdown_price_ratio * eligible_markdown["reference_price"]
    )
    eligible_orders = (
        eligible_markdown[["customer_id", "invoice"]]
        .drop_duplicates()
        .groupby("customer_id", observed=True)
        .size()
    )
    markdown_orders = eligible_markdown.loc[
        eligible_markdown["markdown_like"], ["customer_id", "invoice"]
    ]
    markdown_orders = markdown_orders.drop_duplicates().groupby("customer_id", observed=True).size()
    base["markdown_affinity_proxy"] = (
        base["customer_id"].map(markdown_orders).fillna(0.0)
        / base["customer_id"].map(eligible_orders).replace(0, np.nan)
    ).fillna(0.0)

    encoded = positive.merge(
        artifacts.popularity[["stock_code", "product_popularity", "is_rare_product"]],
        on="stock_code",
        how="left",
    )
    popularity = encoded.groupby("customer_id", observed=True)["product_popularity"].agg(
        mean_product_popularity="mean",
        median_product_popularity="median",
        max_product_popularity="max",
    )
    rare_share = encoded.groupby("customer_id", observed=True)["is_rare_product"].mean()
    base = base.join(popularity, on="customer_id")
    base["rare_product_share"] = base["customer_id"].map(rare_share)
    popularity_columns = [
        "mean_product_popularity",
        "median_product_popularity",
        "max_product_popularity",
        "rare_product_share",
    ]
    base[popularity_columns] = (
        base[popularity_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    affinity = _category_affinity(positive, customer_ids, artifacts)
    base = base.join(affinity, on="customer_id")
    base.insert(1, "cutoff_timestamp", pd.Timestamp(cutoff))
    base = base.drop(columns=["last_purchase", "first_purchase"])
    engagement = EngagementPercentiles.fit(base, training_customer_ids)
    base["engagement_score"] = engagement.transform(base)
    validate_customer_features(base, customer_ids, cutoff)
    return base, engagement


def validate_customer_features(
    features: pd.DataFrame,
    customer_ids: list[str],
    cutoff: pd.Timestamp,
) -> None:
    """Validate feature-table identity, cutoff, numeric finiteness, and affinity sum."""
    if features["customer_id"].duplicated().any() or len(features) != len(customer_ids):
        raise DataValidationError(
            "Customer feature table must contain one row per requested customer"
        )
    if set(features["customer_id"].astype(str)) != set(customer_ids):
        raise DataValidationError("Customer feature population does not match requested snapshots")
    if not features["cutoff_timestamp"].eq(pd.Timestamp(cutoff)).all():
        raise DataValidationError("Customer feature cutoff metadata is inconsistent")
    numeric = features.select_dtypes(include=["number"])
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise DataValidationError("Customer feature table contains non-finite numeric values")
    affinity_columns = [name for name in features if name.startswith("category_affinity_")]
    if affinity_columns and not np.allclose(features[affinity_columns].sum(axis=1), 1.0):
        raise DataValidationError("Category affinity vector must sum to one for eligible customers")
