"""Safe production-model metadata endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["models"])


@router.get("/models/metadata")
def model_metadata(request: Request) -> dict[str, Any]:
    """Expose frozen versions, thresholds, schemas, and genuine evaluation metrics."""
    return request.app.state.artifacts.safe_metadata()
