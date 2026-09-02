"""Service and dependency health endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Return safe database and artifact readiness without exposing configuration."""
    database = "unavailable"
    try:
        with request.app.state.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database = "ready"
    except Exception:  # noqa: BLE001 - health converts dependency failure to safe status
        database = "unavailable"
    artifacts = "ready" if getattr(request.app.state, "artifacts", None) is not None else "missing"
    return {
        "service": "vantara-api",
        "status": "ok" if database == "ready" and artifacts == "ready" else "degraded",
        "database": database,
        "artifacts": artifacts,
    }
