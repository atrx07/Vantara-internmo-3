"""Tests for immutable raw workbook ingestion and validation."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.contracts import WorkbookContract
from src.data.loader import load_transactions, normalize_source_frame
from src.data.validation import (
    DataValidationError,
    validate_canonical_transactions,
    validate_sheet_names,
    validate_source_file,
    validate_source_frame,
)
from src.utils.config import load_config


def _contract(config_path: Path, project_root: Path) -> WorkbookContract:
    return WorkbookContract.from_config(load_config(config_path), project_root=project_root)


def test_load_transactions_combines_sheets_in_chronological_order(
    sample_project: tuple[Path, Path],
) -> None:
    project_root, config_path = sample_project

    frame, summary = load_transactions(config_path, project_root=project_root)

    assert len(frame) == 4
    assert summary.sheet_rows == {"Year 2009-2010": 2, "Year 2010-2011": 2}
    assert summary.combined_rows == 4
    assert summary.chronological is True
    assert frame["invoice_date"].is_monotonic_increasing
    assert frame["invoice"].tolist() == ["10001", "10002", "20001", "20002"]
    assert list(frame.columns) == [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]


def test_normalization_preserves_returns_zero_prices_and_missing_customers(
    sample_project: tuple[Path, Path],
    source_frames: dict[str, pd.DataFrame],
) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)

    normalized = normalize_source_frame(source_frames["Year 2009-2010"], contract)

    assert len(normalized) == 2
    assert normalized.loc[0, "quantity"] == -1
    assert normalized.loc[0, "price"] == 0.0
    assert pd.isna(normalized.loc[0, "customer_id"])
    assert str(normalized["invoice"].dtype).startswith("string")
    assert str(normalized["quantity"].dtype) == "Int64"
    assert str(normalized["invoice_date"].dtype) == "datetime64[ns]"
    assert str(normalized["price"].dtype) == "float64"


def test_source_hash_mismatch_fails(sample_project: tuple[Path, Path]) -> None:
    project_root, config_path = sample_project
    config = load_config(config_path)
    config["data"]["expected_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    contract = _contract(config_path, project_root)

    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        validate_source_file(contract)


def test_missing_source_workbook_fails(sample_project: tuple[Path, Path]) -> None:
    project_root, config_path = sample_project
    config = load_config(config_path)
    config["data"]["raw_workbook"] = "data/raw/missing.xlsx"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    contract = _contract(config_path, project_root)

    with pytest.raises(DataValidationError, match="does not exist"):
        validate_source_file(contract)


def test_sheet_name_order_mismatch_fails(sample_project: tuple[Path, Path]) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)

    with pytest.raises(DataValidationError, match="sheets mismatch"):
        validate_sheet_names(["Year 2010-2011", "Year 2009-2010"], contract)


def test_source_column_mismatch_fails(
    sample_project: tuple[Path, Path],
    source_frames: dict[str, pd.DataFrame],
) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)
    malformed = source_frames["Year 2009-2010"].drop(columns=["Country"])

    with pytest.raises(DataValidationError, match="Source columns mismatch"):
        validate_source_frame(malformed, sheet_name="Year 2009-2010", contract=contract)


def test_source_row_count_mismatch_fails(
    sample_project: tuple[Path, Path],
    source_frames: dict[str, pd.DataFrame],
) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)
    shortened = source_frames["Year 2009-2010"].iloc[:1]

    with pytest.raises(DataValidationError, match="Row count mismatch"):
        validate_source_frame(shortened, sheet_name="Year 2009-2010", contract=contract)


def test_null_rate_threshold_fails(sample_project: tuple[Path, Path]) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)
    frame, _ = load_transactions(config_path, project_root=project_root)
    frame["description"] = pd.Series([pd.NA] * len(frame), dtype="string")

    with pytest.raises(DataValidationError, match="Null rate for description"):
        validate_canonical_transactions(frame, contract)


def test_non_integral_quantity_fails(
    sample_project: tuple[Path, Path],
    source_frames: dict[str, pd.DataFrame],
) -> None:
    project_root, config_path = sample_project
    contract = _contract(config_path, project_root)
    malformed = source_frames["Year 2009-2010"].copy()
    malformed["Quantity"] = malformed["Quantity"].astype("float64")
    malformed.loc[0, "Quantity"] = 1.5

    with pytest.raises(DataValidationError, match="non-integral"):
        normalize_source_frame(malformed, contract)
