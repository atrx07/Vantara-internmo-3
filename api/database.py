"""SQLAlchemy engine and session configuration for the serving API."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from fastapi import Request
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry


def database_url_from_environment() -> str:
    """Return DATABASE_URL without embedding a password or development fallback."""
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required to start the Vantara API")
    return value


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy 2.x engine with safe liveness checking."""
    arguments: dict[str, Any] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        arguments["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, **arguments)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(
            connection: DBAPIConnection, _connection_record: ConnectionPoolEntry
        ) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the non-expiring transactional session factory used by requests."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session(request: Request) -> Generator[Session, None, None]:
    """Yield one request-scoped SQLAlchemy session."""
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        yield session
