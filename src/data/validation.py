"""Schema, hash, row-count, null-rate, and date validation for raw ingestion."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.data.contracts import WorkbookContract
from src.utils.hashing import sha256_file


class DataValidationError(ValueError):
    """Raised when immutable source data violates its governed contract."""


def validate_source_file(contract: WorkbookContract) -> str:
    """Require the configured workbook to exist and match its exact SHA-256 digest."""
    path = Path(contract.path)
    if not path.is_file():
        raise DataValidationError(f"Raw workbook does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != contract.expected_sha256:
        raise DataValidationError(
            f"Raw workbook SHA-256 mismatch: expected {contract.expected_sha256}, "
            f"got {actual_hash}"
        )
    return actual_hash


def validate_sheet_names(actual: list[str], contract: WorkbookContract) -> None:
    """Require workbook sheet names and order to match the manifest exactly."""
    expected = list(contract.sheets)
    if actual != expected:
        raise DataValidationError(f"Workbook sheets mismatch: expected {expected}, got {actual}")


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    sheet_name: str,
    contract: WorkbookContract,
) -> None:
    """Validate one unmodified source sheet's columns and exact data-row count."""
    actual_columns = list(frame.columns)
    expected_columns = list(contract.source_columns)
    if actual_columns != expected_columns:
        raise DataValidationError(
            f"Source columns mismatch in {sheet_name}: expected {expected_columns}, "
            f"got {actual_columns}"
        )
    expected_rows = contract.sheets[sheet_name]
    if len(frame) != expected_rows:
        raise DataValidationError(
            f"Row count mismatch in {sheet_name}: expected {expected_rows}, got {len(frame)}"
        )


def canonical_transaction_schema() -> pa.DataFrameSchema:
    """Return the strict Pandera schema for the STEP 01 canonical transaction table."""
    return pa.DataFrameSchema(
        {
            "invoice": pa.Column(pa.String, nullable=False),
            "stock_code": pa.Column(pa.String, nullable=False),
            "description": pa.Column(pa.String, nullable=True),
            "quantity": pa.Column(pa.Int64, nullable=False),
            "invoice_date": pa.Column(pa.DateTime, nullable=False),
            "price": pa.Column(pa.Float64, nullable=False),
            "customer_id": pa.Column(pa.String, nullable=True),
            "country": pa.Column(pa.String, nullable=False),
        },
        strict=True,
        ordered=True,
        coerce=False,
    )


def validate_canonical_transactions(
    frame: pd.DataFrame,
    contract: WorkbookContract,
) -> None:
    """Validate canonical types, null rates, date range, and chronological ordering."""
    try:
        canonical_transaction_schema().validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise DataValidationError(f"Canonical transaction schema failed: {exc}") from exc

    for column, maximum_rate in contract.maximum_null_rates.items():
        actual_rate = float(frame[column].isna().mean())
        if actual_rate > maximum_rate:
            raise DataValidationError(
                f"Null rate for {column} exceeds threshold: "
                f"{actual_rate:.6f} > {maximum_rate:.6f}"
            )

    minimum_date = frame["invoice_date"].min()
    maximum_date = frame["invoice_date"].max()
    if minimum_date != contract.minimum_date or maximum_date != contract.maximum_date:
        raise DataValidationError(
            "Invoice date range mismatch: "
            f"expected {contract.minimum_date} through {contract.maximum_date}, "
            f"got {minimum_date} through {maximum_date}"
        )
    if not frame["invoice_date"].is_monotonic_increasing:
        raise DataValidationError("Combined transactions are not chronologically ordered")
