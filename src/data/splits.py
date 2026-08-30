"""Deterministic persisted customer-level partition contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validation import DataValidationError

PARTITIONS = ("train", "validation", "test")


def create_customer_split(
    customer_ids: list[str],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    version: str,
) -> pd.DataFrame:
    """Create one deterministic 70/15/15 customer partition table."""
    if not customer_ids:
        raise DataValidationError("Cannot split an empty customer population")
    if len(set(customer_ids)) != len(customer_ids):
        raise DataValidationError("Customer split input contains duplicate identifiers")
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise DataValidationError("Customer split fractions must sum to one")

    ordered = np.asarray(sorted(str(value) for value in customer_ids), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    train_end = int(np.floor(len(shuffled) * train_fraction))
    validation_end = train_end + int(np.floor(len(shuffled) * validation_fraction))
    partition = np.full(len(shuffled), "test", dtype=object)
    partition[:train_end] = "train"
    partition[train_end:validation_end] = "validation"
    result = pd.DataFrame(
        {
            "customer_id": pd.Series(shuffled, dtype="string"),
            "partition": pd.Categorical(partition, categories=list(PARTITIONS)),
            "split_seed": np.int64(seed),
            "split_version": pd.Series([version] * len(shuffled), dtype="string"),
        }
    ).sort_values("customer_id", kind="mergesort", ignore_index=True)
    validate_customer_split(result)
    return result


def validate_customer_split(split: pd.DataFrame) -> None:
    """Require unique customers and all three mutually exclusive partitions."""
    required = {"customer_id", "partition", "split_seed", "split_version"}
    if not required.issubset(split.columns):
        raise DataValidationError(
            f"Customer split missing columns: {sorted(required - set(split.columns))}"
        )
    if split["customer_id"].isna().any() or split["customer_id"].duplicated().any():
        raise DataValidationError("Each identified customer must occur exactly once in the split")
    actual = set(split["partition"].astype("string").dropna().unique())
    if actual != set(PARTITIONS):
        raise DataValidationError(f"Customer split partitions mismatch: {sorted(actual)}")
    groups = {
        name: set(split.loc[split["partition"] == name, "customer_id"].astype(str))
        for name in PARTITIONS
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"]:
        raise DataValidationError("Customer split partitions overlap")
    if groups["validation"] & groups["test"]:
        raise DataValidationError("Customer split partitions overlap")


def customer_ids_for_partition(split: pd.DataFrame, partition: str) -> set[str]:
    """Return the customer identifiers assigned to one validated partition."""
    if partition not in PARTITIONS:
        raise DataValidationError(f"Unknown partition: {partition}")
    validate_customer_split(split)
    return set(split.loc[split["partition"] == partition, "customer_id"].astype(str))
