"""Single-command deterministic raw-to-frozen-feature M1 pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.analysis.eda import run_step03_analysis
from src.data.step02_pipeline import run_step02
from src.data.validation import DataValidationError
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Run immutable ingestion through cleaned, featured, analyzed, frozen M1 outputs."""
    step02 = run_step02(config_path)
    step03 = run_step03_analysis(config_path)
    return {"step02": step02, "step03": step03}


def build_parser() -> argparse.ArgumentParser:
    """Build the M1 pipeline command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the complete M1 data pipeline with structured logging."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    try:
        result = run_pipeline(args.config)
    except (DataValidationError, OSError, ValueError, KeyError) as exc:
        LOGGER.error(
            "M1 pipeline failed",
            extra={"event": "m1_pipeline_failed"},
            exc_info=exc,
        )
        return 1
    LOGGER.info(
        "M1 pipeline passed",
        extra={
            "event": "m1_pipeline_passed",
            "rows": int(result["step03"]["customer_rows"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
