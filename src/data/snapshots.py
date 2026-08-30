"""Point-in-time customer snapshot and governed target-window abstractions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.validation import DataValidationError


@dataclass(frozen=True, order=True)
class CustomerSnapshot:
    """One customer observed strictly before a prediction cutoff timestamp."""

    customer_id: str
    cutoff_timestamp: pd.Timestamp

    def history(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return this customer's identified rows strictly before the cutoff."""
        return transactions.loc[
            transactions["customer_id"].eq(self.customer_id)
            & transactions["invoice_date"].lt(self.cutoff_timestamp)
        ].copy()

    def target_window(self, transactions: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        """Return rows in the fully observable [cutoff, cutoff+horizon] window."""
        end = self.cutoff_timestamp + pd.Timedelta(days=horizon_days)
        return transactions.loc[
            transactions["customer_id"].eq(self.customer_id)
            & transactions["invoice_date"].ge(self.cutoff_timestamp)
            & transactions["invoice_date"].le(end)
        ].copy()


@dataclass(frozen=True)
class CanonicalCutoffs:
    """Observation end and derived churn/CLV cutoffs."""

    observation_end: pd.Timestamp
    churn_cutoff: pd.Timestamp
    clv_cutoff: pd.Timestamp


def derive_canonical_cutoffs(
    transactions: pd.DataFrame,
    *,
    churn_horizon_days: int,
    clv_horizon_days: int,
) -> CanonicalCutoffs:
    """Derive canonical prediction cutoffs from the actual maximum source timestamp."""
    if transactions.empty:
        raise DataValidationError("Cannot derive cutoffs from empty transactions")
    observation_end = pd.Timestamp(transactions["invoice_date"].max())
    churn_cutoff = observation_end - pd.Timedelta(days=churn_horizon_days)
    clv_cutoff = observation_end - pd.Timedelta(days=clv_horizon_days)
    if clv_cutoff >= churn_cutoff:
        raise DataValidationError("CLV cutoff must precede churn cutoff")
    return CanonicalCutoffs(observation_end, churn_cutoff, clv_cutoff)


def eligible_customer_ids(
    transactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> list[str]:
    """Return sorted customers with a valid positive purchase before the cutoff."""
    eligible = transactions.loc[
        transactions["invoice_date"].lt(cutoff)
        & transactions["is_positive_purchase"]
        & transactions["customer_id"].notna(),
        "customer_id",
    ]
    return sorted(str(value) for value in eligible.unique())


def churn_labels(
    transactions: pd.DataFrame,
    *,
    customer_ids: list[str],
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    """Create the canonical no-positive-purchase-in-90-days churn label."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    future_buyers = set(
        transactions.loc[
            transactions["customer_id"].isin(customer_ids)
            & transactions["invoice_date"].ge(cutoff)
            & transactions["invoice_date"].le(end)
            & transactions["is_positive_purchase"],
            "customer_id",
        ].astype(str)
    )
    return pd.DataFrame(
        {
            "customer_id": pd.Series(customer_ids, dtype="string"),
            "churn": pd.Series(
                [0 if customer_id in future_buyers else 1 for customer_id in customer_ids],
                dtype="int8",
            ),
        }
    )


def clv_targets(
    transactions: pd.DataFrame,
    *,
    customer_ids: list[str],
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    """Create clipped 180-day forward net-merchandise-revenue proxy targets."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    future = transactions.loc[
        transactions["customer_id"].isin(customer_ids)
        & transactions["invoice_date"].ge(cutoff)
        & transactions["invoice_date"].le(end)
        & transactions["is_valid_merchandise"],
        ["customer_id", "signed_merchandise_value"],
    ]
    totals = future.groupby("customer_id", observed=True)["signed_merchandise_value"].sum()
    values = [max(float(totals.get(customer_id, 0.0)), 0.0) for customer_id in customer_ids]
    return pd.DataFrame(
        {
            "customer_id": pd.Series(customer_ids, dtype="string"),
            "clv_180d_target": pd.Series(values, dtype="float64"),
        }
    )
