"""Deterministic plain-language churn explanation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def plain_language_churn_explanation(
    *,
    probability: float,
    threshold: float,
    positive_drivers: Sequence[str],
    negative_drivers: Sequence[str],
) -> str:
    """Render a deterministic business-readable explanation from signed contributions."""
    decision = "above" if probability >= threshold else "below"
    outcome = "prioritize for retention review" if probability >= threshold else "monitor normally"
    upward = ", ".join(positive_drivers[:3]) if positive_drivers else "no dominant upward driver"
    downward = (
        ", ".join(negative_drivers[:3]) if negative_drivers else "no dominant downward driver"
    )
    return (
        f"Estimated churn risk is {probability:.1%}, {decision} the frozen {threshold:.1%} "
        f"decision threshold; {outcome}. Main upward drivers: {upward}. "
        f"Main risk-reducing drivers: {downward}."
    )
