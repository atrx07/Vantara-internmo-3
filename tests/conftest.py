"""Reusable STEP 01 test fixtures."""

import os
from collections.abc import Generator
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


@pytest.fixture()
def cleaning_config() -> dict[str, object]:
    """Return a compact governed cleaning configuration for unit tests."""
    return {
        "administrative_stock_codes": ["POST", "M", "D"],
        "administrative_stock_code_patterns": ["^TEST", "^GIFT_"],
        "outliers": {
            "iqr_multiplier": 3.0,
            "quantity_absolute_domain_limit": 100000,
            "price_domain_limit": 50000.0,
        },
    }


@pytest.fixture()
def step02_transactions(cleaning_config: dict[str, object]) -> pd.DataFrame:
    """Create identified multi-customer transactions suitable for STEP 02 tests."""
    from src.data.cleaning import clean_transactions

    records: list[list[object]] = []
    products = [f"P{index:02d}" for index in range(12)]
    for customer_index in range(8):
        customer_id = f"CUST{customer_index}"
        for order_index in range(6):
            timestamp = pd.Timestamp("2021-01-05") + pd.Timedelta(
                days=customer_index + order_index * 25
            )
            stock_code = products[(customer_index + order_index) % len(products)]
            records.append(
                [
                    f"{customer_index}{order_index:03d}",
                    stock_code,
                    f"Product family {stock_code}",
                    order_index + 1,
                    timestamp,
                    10.0 + (order_index % 3),
                    customer_id,
                    "United Kingdom",
                ]
            )
    records.extend(
        [
            [
                "C900",
                "P00",
                "Product family P00",
                -1,
                "2021-04-01",
                10.0,
                "CUST0",
                "United Kingdom",
            ],
            ["9001", "POST", "Postage", 1, "2021-03-01", 5.0, "CUST0", "United Kingdom"],
            ["9002", "P01", "  product   FAMILY p01 ", 1, "2021-03-02", 0.0, None, "France"],
        ]
    )
    frame = pd.DataFrame(
        records,
        columns=[
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "price",
            "customer_id",
            "country",
        ],
    )
    frame["invoice"] = frame["invoice"].astype("string")
    frame["stock_code"] = frame["stock_code"].astype("string")
    frame["description"] = frame["description"].astype("string")
    frame["quantity"] = frame["quantity"].astype("Int64")
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["price"] = frame["price"].astype("float64")
    frame["customer_id"] = frame["customer_id"].astype("string")
    frame["country"] = frame["country"].astype("string")
    cleaned, _ = clean_transactions(frame, cleaning_config=cleaning_config)
    return cleaned


@pytest.fixture(scope="session")
def migrated_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[dict[str, object], None, None]:
    """Create, migrate, and seed one fresh disposable STEP 07 serving database."""
    from alembic import command
    from alembic.config import Config

    from api.artifacts import ArtifactRegistry
    from api.database import create_database_engine
    from api.initialization import initialize_serving_data
    from src.utils.config import load_config

    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path_factory.mktemp("step07") / "serving.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        alembic = Config(str(root / "alembic.ini"))
        command.upgrade(alembic, "head")
        engine = create_database_engine(database_url)
        registry = ArtifactRegistry(root, root / "models_artifacts")
        config = load_config(root / "config" / "config.yaml")
        counts = initialize_serving_data(engine, root, registry, config)
        yield {
            "root": root,
            "database_url": database_url,
            "engine": engine,
            "registry": registry,
            "config": config,
            "counts": counts,
        }
        engine.dispose()
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
