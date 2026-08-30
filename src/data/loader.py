"""Read-only Online Retail II workbook ingestion and canonical normalization."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.contracts import IngestionSummary, WorkbookContract
from src.data.validation import (
    DataValidationError,
    validate_canonical_transactions,
    validate_sheet_names,
    validate_source_file,
    validate_source_frame,
)
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def normalize_source_frame(frame: pd.DataFrame, contract: WorkbookContract) -> pd.DataFrame:
    """Map source columns and types without cleaning, filtering, or changing row count."""
    normalized = frame.rename(columns=contract.canonical_columns).copy()

    for column in ("invoice", "stock_code", "description", "customer_id", "country"):
        normalized[column] = normalized[column].astype("string").str.strip()
        normalized[column] = normalized[column].mask(normalized[column].eq(""), pd.NA)

    quantity = pd.to_numeric(normalized["quantity"], errors="raise")
    non_integral = quantity.notna() & quantity.mod(1).ne(0)
    if bool(non_integral.any()):
        raise DataValidationError("Quantity contains non-integral source values")
    normalized["quantity"] = quantity.astype("Int64")
    normalized["invoice_date"] = pd.to_datetime(normalized["invoice_date"], errors="raise")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="raise").astype("float64")

    canonical_order = list(contract.canonical_columns.values())
    return normalized.loc[:, canonical_order]


def load_transactions(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> tuple[pd.DataFrame, IngestionSummary]:
    """Load, normalize, combine, sort, and validate both immutable workbook sheets."""
    config = load_config(config_path)
    contract = WorkbookContract.from_config(config, project_root=project_root)
    actual_hash = validate_source_file(contract)

    frames: list[pd.DataFrame] = []
    sheet_rows: dict[str, int] = {}
    string_columns = {
        "Invoice": "string",
        "StockCode": "string",
        "Description": "string",
        "Customer ID": "string",
        "Country": "string",
    }

    with pd.ExcelFile(contract.path, engine="openpyxl") as workbook:
        validate_sheet_names(workbook.sheet_names, contract)
        for sheet_name in contract.sheets:
            source = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                dtype=string_columns,
                engine="openpyxl",
            )
            validate_source_frame(source, sheet_name=sheet_name, contract=contract)
            canonical = normalize_source_frame(source, contract)
            if len(canonical) != len(source):
                raise DataValidationError(
                    f"Normalization changed row count for {sheet_name}: "
                    f"{len(source)} -> {len(canonical)}"
                )
            frames.append(canonical)
            sheet_rows[sheet_name] = len(canonical)
            LOGGER.info(
                "Loaded raw workbook sheet",
                extra={"event": "raw_sheet_loaded", "sheet": sheet_name, "rows": len(canonical)},
            )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)
    validate_canonical_transactions(combined, contract)

    summary = IngestionSummary(
        path=str(contract.path),
        sha256=actual_hash,
        sheet_rows=sheet_rows,
        combined_rows=len(combined),
        columns=tuple(str(column) for column in combined.columns),
        minimum_date=combined["invoice_date"].min().isoformat(sep=" "),
        maximum_date=combined["invoice_date"].max().isoformat(sep=" "),
        chronological=bool(combined["invoice_date"].is_monotonic_increasing),
    )
    return combined, summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the raw ingestion validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the governed read-only raw ingestion validation command."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    try:
        _, summary = load_transactions(args.config)
    except (DataValidationError, OSError, ValueError) as exc:
        LOGGER.error(
            "Raw ingestion validation failed",
            extra={"event": "raw_ingestion_failed"},
            exc_info=exc,
        )
        return 1

    LOGGER.info(
        "Raw ingestion validation passed",
        extra={
            "event": "raw_ingestion_passed",
            "path": summary.path,
            "rows": summary.combined_rows,
            "columns": len(summary.columns),
            "minimum_date": summary.minimum_date,
            "maximum_date": summary.maximum_date,
            "sha256": summary.sha256,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
