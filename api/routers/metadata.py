"""Safe production-model metadata endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["models"])


@router.get("/models/metadata")
def model_metadata(request: Request) -> dict[str, Any]:
    """Expose frozen versions, thresholds, schemas, and genuine evaluation metrics."""
    return request.app.state.artifacts.safe_metadata()


@router.get("/models/insights/churn-pdp")
def churn_partial_dependence(request: Request) -> dict[str, str]:
    """Return the tracked churn partial-dependence visual as base64 PNG data."""
    return {"image_base64": request.app.state.artifacts.churn_partial_dependence_base64}
