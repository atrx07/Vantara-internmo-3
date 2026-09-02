"""Initialize deterministic Vantara serving state after Alembic migration."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from api.artifacts import ArtifactRegistry
from api.database import create_database_engine, database_url_from_environment
from api.initialization import initialize_serving_data
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build serving-data initialization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--include-transactions", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load deterministic serving state without retraining any model."""
    arguments = build_parser().parse_args(argv)
    root = Path.cwd().resolve()
    config = load_config(arguments.config)
    configure_logging(os.getenv("LOG_LEVEL", str(config["logging"]["level"])))
    artifact_root = Path(os.getenv("MODEL_ARTIFACT_DIR", "models_artifacts"))
    if not artifact_root.is_absolute():
        artifact_root = root / artifact_root
    registry = ArtifactRegistry(root, artifact_root)
    engine = create_database_engine(database_url_from_environment())
    counts = initialize_serving_data(
        engine,
        root,
        registry,
        config,
        include_transactions=arguments.include_transactions,
        replace=arguments.replace,
    )
    LOGGER.info("Serving data initialized", extra={"event": "serving_data_initialized", **counts})
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
