"""FastAPI application factory with lifespan-owned artifacts and database state."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import text

from api.artifacts import ArtifactRegistry
from api.database import (
    create_database_engine,
    create_session_factory,
    database_url_from_environment,
)
from api.routers import customers, health, metadata, predictions
from src.utils.config import load_config
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    database_url: str | None = None,
    project_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> FastAPI:
    """Create the Vantara API with explicit test/runtime overrides and no secret defaults."""
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    config = load_config(root / "config" / "config.yaml")
    configure_logging(os.getenv("LOG_LEVEL", str(config["logging"]["level"])))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        url = database_url or database_url_from_environment()
        configured_artifact_root = artifact_root or os.getenv(
            "MODEL_ARTIFACT_DIR", "models_artifacts"
        )
        artifacts = Path(configured_artifact_root)
        if not artifacts.is_absolute():
            artifacts = root / artifacts
        engine = create_database_engine(url)
        registry = ArtifactRegistry(root, artifacts)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        application.state.artifacts = registry
        LOGGER.info(
            "Vantara API dependencies loaded",
            extra={"event": "api_startup_complete", "database_dialect": engine.dialect.name},
        )
        try:
            yield
        finally:
            engine.dispose()

    application = FastAPI(
        title="Vantara Customer Behavior API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.project_root = root
    application.state.config = config
    prefix = str(config["serving"]["api_prefix"])
    for router in (health.router, metadata.router, predictions.router, customers.router):
        application.include_router(router, prefix=prefix)
    return application


app = create_app()
