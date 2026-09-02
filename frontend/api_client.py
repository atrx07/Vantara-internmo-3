"""Typed HTTP boundary used by Streamlit to consume the Vantara FastAPI service."""

from __future__ import annotations

from typing import Any, cast

import httpx


class APIClientError(RuntimeError):
    """User-safe API communication or validation error."""


class DashboardAPI:
    """Small synchronous client for the dashboard's governed API surface."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized = f"{base_url.rstrip('/')}/"
        self._client = httpx.Client(
            base_url=normalized,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json: dict[str, object] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            response = self._client.request(
                method,
                path.lstrip("/"),
                params=params,
                json=json,
                files=files,
            )
            response.raise_for_status()
            return cast(dict[str, Any] | list[dict[str, Any]], response.json())
        except httpx.HTTPStatusError as error:
            try:
                detail = error.response.json().get("detail", error.response.text)
            except (ValueError, AttributeError):
                detail = error.response.text
            raise APIClientError(
                str(detail) or "The Vantara service rejected the request"
            ) from error
        except (httpx.RequestError, ValueError) as error:
            raise APIClientError(
                "The Vantara API is unavailable. Confirm that the API and database are running."
            ) from error

    def health(self) -> dict[str, Any]:
        """Return safe service dependency health."""
        return dict(self._request("GET", "health"))

    def overview(self) -> dict[str, Any]:
        """Return executive overview metrics."""
        return dict(self._request("GET", "analytics/overview"))

    def segments(self) -> list[dict[str, Any]]:
        """Return all current segment summaries."""
        return list(self._request("GET", "segments"))

    def segment_customers(
        self,
        segment_id: int,
        *,
        country: str | None = None,
        value_tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return customers in a segment after optional business filters."""
        params = {"country": country, "value_tier": value_tier, "limit": 5000}
        return list(
            self._request(
                "GET",
                f"segments/{segment_id}",
                params={key: value for key, value in params.items() if value is not None},
            )
        )

    def churn_priority(
        self,
        *,
        segment_id: int | None = None,
        country: str | None = None,
        value_tier: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        """Return the latest scored customers ordered by locked retention priority."""
        params = {
            "segment_id": segment_id,
            "country": country,
            "value_tier": value_tier,
            "limit": limit,
        }
        return list(
            self._request(
                "GET",
                "analytics/churn-priority",
                params={key: value for key, value in params.items() if value is not None},
            )
        )

    def customer(self, customer_id: str) -> dict[str, Any]:
        """Return a persisted customer summary."""
        return dict(self._request("GET", f"customers/{customer_id}"))

    def score_customer(self, customer_id: str) -> dict[str, Any]:
        """Score and persist one known customer."""
        return dict(self._request("POST", "predict/customer", json={"customer_id": customer_id}))

    def explanation(self, customer_id: str) -> dict[str, Any]:
        """Return individual TreeSHAP drivers and plain-language explanation."""
        return dict(self._request("GET", f"customers/{customer_id}/explanation"))

    def recommendations(self, customer_id: str) -> list[dict[str, Any]]:
        """Return ranked product recommendations for one customer."""
        return list(self._request("GET", f"customers/{customer_id}/recommendations"))

    def revenue(self) -> list[dict[str, Any]]:
        """Return persisted monthly historical net revenue."""
        return list(self._request("GET", "analytics/revenue"))

    def metadata(self) -> dict[str, Any]:
        """Return safe frozen-model metadata and held-out metrics."""
        return dict(self._request("GET", "models/metadata"))

    def model_insights(self) -> dict[str, Any]:
        """Return global explainability and model-comparison evidence."""
        return dict(self._request("GET", "models/insights"))

    def churn_partial_dependence(self) -> str:
        """Return the tracked partial-dependence PNG encoded for safe transport."""
        payload = dict(self._request("GET", "models/insights/churn-pdp"))
        return str(payload["image_base64"])

    def score_batch(self, filename: str, content: bytes) -> list[dict[str, Any]]:
        """Upload canonical transaction history and return persisted customer scores."""
        files = {"file": (filename, content, "text/csv")}
        return list(self._request("POST", "predict/batch", files=files))
