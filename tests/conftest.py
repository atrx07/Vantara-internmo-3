"""Reusable STEP 01 test fixtures."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.utils.hashing import sha256_file


@pytest.fixture()
def source_frames() -> dict[str, pd.DataFrame]:
    """Return two small Online Retail II-shaped source sheets."""
    columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    first = pd.DataFrame(
        [
            ["10002", "B", "Second", -1, "2009-12-02 09:00:00", 0.0, None, "United Kingdom"],
            ["10001", "A", "First", 2, "2009-12-01 07:45:00", 1.5, 12345, "France"],
        ],
        columns=columns,
    )
    second = pd.DataFrame(
        [
            ["20002", "D", None, 3, "2011-12-09 12:50:00", 4.0, 67890, "Germany"],
            ["20001", "C", "Third", 1, "2010-12-01 08:26:00", 2.5, 12345, "France"],
        ],
        columns=columns,
    )
    return {"Year 2009-2010": first, "Year 2010-2011": second}


@pytest.fixture()
def sample_project(tmp_path: Path, source_frames: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    """Create a temporary two-sheet workbook and matching YAML contract."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    workbook_path = raw_dir / "online_retail_II.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in source_frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    config = {
        "project": {"name": "test", "python_version": "3.11", "random_seed": 42},
        "logging": {"level": "INFO"},
        "data": {
            "raw_workbook": "data/raw/online_retail_II.xlsx",
            "expected_sha256": sha256_file(workbook_path),
            "sheets": {name: len(frame) for name, frame in source_frames.items()},
            "source_columns": list(next(iter(source_frames.values())).columns),
            "canonical_columns": {
                "Invoice": "invoice",
                "StockCode": "stock_code",
                "Description": "description",
                "Quantity": "quantity",
                "InvoiceDate": "invoice_date",
                "Price": "price",
                "Customer ID": "customer_id",
                "Country": "country",
            },
            "expected_date_range": {
                "minimum": "2009-12-01 07:45:00",
                "maximum": "2011-12-09 12:50:00",
            },
            "maximum_null_rates": {
                "invoice": 0.0,
                "stock_code": 0.0,
                "description": 0.25,
                "quantity": 0.0,
                "invoice_date": 0.0,
                "price": 0.0,
                "customer_id": 0.25,
                "country": 0.0,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return tmp_path, config_path
