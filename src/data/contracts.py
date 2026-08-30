"""Typed workbook and ingestion contracts for the immutable raw source."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ConfigurationError, resolve_project_path


@dataclass(frozen=True)
class WorkbookContract:
    """Immutable expectations for the supplied Online Retail II workbook."""

    path: Path
    expected_sha256: str
    sheets: dict[str, int]
    source_columns: tuple[str, ...]
    canonical_columns: dict[str, str]
    minimum_date: pd.Timestamp
    maximum_date: pd.Timestamp
    maximum_null_rates: dict[str, float]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        project_root: str | Path | None = None,
    ) -> "WorkbookContract":
        """Build and validate a workbook contract from the project YAML mapping."""
        try:
            data = config["data"]
            date_range = data["expected_date_range"]
            sheets = {str(name): int(rows) for name, rows in data["sheets"].items()}
            source_columns = tuple(str(name) for name in data["source_columns"])
            canonical_columns = {
                str(source): str(target) for source, target in data["canonical_columns"].items()
            }
            maximum_null_rates = {
                str(column): float(limit) for column, limit in data["maximum_null_rates"].items()
            }
            contract = cls(
                path=resolve_project_path(data["raw_workbook"], project_root=project_root),
                expected_sha256=str(data["expected_sha256"]).lower(),
                sheets=sheets,
                source_columns=source_columns,
                canonical_columns=canonical_columns,
                minimum_date=pd.Timestamp(date_range["minimum"]),
                maximum_date=pd.Timestamp(date_range["maximum"]),
                maximum_null_rates=maximum_null_rates,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid data contract configuration: {exc}") from exc

        if not contract.sheets:
            raise ConfigurationError("At least one workbook sheet must be configured")
        if tuple(contract.canonical_columns) != contract.source_columns:
            raise ConfigurationError(
                "canonical_columns keys must preserve the configured source_columns order"
            )
        if len(set(contract.canonical_columns.values())) != len(contract.canonical_columns):
            raise ConfigurationError("Canonical column names must be unique")
        for column, limit in contract.maximum_null_rates.items():
            if not 0.0 <= limit <= 1.0:
                raise ConfigurationError(f"maximum_null_rates.{column} must be between 0 and 1")
        return contract


@dataclass(frozen=True)
class IngestionSummary:
    """Auditable summary of a successfully loaded canonical transaction table."""

    path: str
    sha256: str
    sheet_rows: dict[str, int]
    combined_rows: int
    columns: tuple[str, ...]
    minimum_date: str
    maximum_date: str
    chronological: bool
