"""Cleaning, quality-flag, cutoff, label, and split contract tests."""

import pandas as pd

from src.data.cleaning import clean_transactions, fit_outlier_thresholds
from src.data.snapshots import (
    CustomerSnapshot,
    churn_labels,
    clv_targets,
    derive_canonical_cutoffs,
)
from src.data.splits import create_customer_split


def test_cleaning_removes_only_exact_duplicates_and_flags_quality(
    step02_transactions: pd.DataFrame,
    cleaning_config: dict[str, object],
) -> None:
    source = step02_transactions.iloc[:3].copy()
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    legitimate_repeat = source.iloc[[0]].copy()
    legitimate_repeat["quantity"] = legitimate_repeat["quantity"] + 1
    source = pd.concat([duplicate, legitimate_repeat], ignore_index=True)

    cleaned, summary = clean_transactions(source, cleaning_config=cleaning_config)

    assert summary.exact_duplicates_removed == 1
    assert len(cleaned) == 4
    assert cleaned["description_normalized"].str.match(r"^[A-Z0-9 ]+$").all()


def test_return_cancel_admin_price_and_missing_id_flags(
    step02_transactions: pd.DataFrame,
) -> None:
    cancelled = step02_transactions.loc[step02_transactions["invoice"] == "C900"].iloc[0]
    admin = step02_transactions.loc[step02_transactions["stock_code"] == "POST"].iloc[0]
    missing = step02_transactions.loc[step02_transactions["customer_id"].isna()].iloc[0]

    assert cancelled["is_return"]
    assert cancelled["is_cancelled_invoice"]
    assert not cancelled["is_positive_purchase"]
    assert admin["is_administrative_line"]
    assert not admin["is_product"]
    assert missing["is_missing_customer_id"]
    assert missing["is_non_positive_price"]


def test_outlier_thresholds_fit_training_customers_only(
    step02_transactions: pd.DataFrame,
    cleaning_config: dict[str, object],
) -> None:
    training_ids = {"CUST0", "CUST1", "CUST2", "CUST3"}
    baseline = fit_outlier_thresholds(
        step02_transactions,
        training_customer_ids=training_ids,
        config=cleaning_config["outliers"],  # type: ignore[arg-type]
    )
    attack = step02_transactions.copy()
    validation_row = attack.loc[attack["customer_id"] == "CUST7"].index[0]
    attack.loc[validation_row, "price"] = 49999.0
    attacked = fit_outlier_thresholds(
        attack,
        training_customer_ids=training_ids,
        config=cleaning_config["outliers"],  # type: ignore[arg-type]
    )

    assert baseline == attacked


def test_customer_snapshot_history_is_strictly_before_cutoff(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-03-01")
    snapshot = CustomerSnapshot("CUST0", cutoff)
    history = snapshot.history(step02_transactions)

    assert (history["invoice_date"] < cutoff).all()
    assert not (history["invoice_date"] == cutoff).any()


def test_canonical_cutoffs_are_derived_from_observation_end(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoffs = derive_canonical_cutoffs(
        step02_transactions, churn_horizon_days=90, clv_horizon_days=180
    )

    assert cutoffs.observation_end == step02_transactions["invoice_date"].max()
    assert cutoffs.churn_cutoff == cutoffs.observation_end - pd.Timedelta(days=90)
    assert cutoffs.clv_cutoff == cutoffs.observation_end - pd.Timedelta(days=180)


def test_churn_ignores_returns_and_clv_uses_only_target_window(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-03-20")
    customers = ["CUST0", "CUST1"]
    churn = churn_labels(
        step02_transactions,
        customer_ids=customers,
        cutoff=cutoff,
        horizon_days=30,
    ).set_index("customer_id")
    clv = clv_targets(
        step02_transactions,
        customer_ids=customers,
        cutoff=cutoff,
        horizon_days=30,
    ).set_index("customer_id")

    assert churn["churn"].to_dict() == {"CUST0": 0, "CUST1": 0}
    assert clv["clv_180d_target"].to_dict() == {"CUST0": 85.0, "CUST1": 95.0}


def test_customer_split_is_deterministic_disjoint_and_approximately_70_15_15() -> None:
    customers = [f"CUST{index:03d}" for index in range(100)]
    first = create_customer_split(
        customers,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        version="test-v1",
    )
    second = create_customer_split(
        customers,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        version="test-v1",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["customer_id"].nunique() == 100
    assert first["partition"].value_counts().to_dict() == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
