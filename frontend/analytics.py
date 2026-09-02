"""Pure dashboard transformations for display names and revenue forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass(frozen=True)
class ForecastResult:
    """Historical and forecast frames plus the disclosed method label."""

    historical: pd.DataFrame
    forecast: pd.DataFrame
    method: str


def friendly_feature_name(value: str) -> str:
    """Convert a technical feature key into a compact business-facing label."""
    replacements = {
        "clv": "customer value",
        "rfm": "customer activity",
        "shap": "impact",
    }
    words = value.replace("_", " ").strip().lower()
    for source, target in replacements.items():
        words = words.replace(source, target)
    return words.title()


def revenue_forecast(
    points: list[dict[str, object]],
    *,
    periods: int = 6,
    seasonal_periods: int = 12,
) -> ForecastResult:
    """Forecast monthly revenue with Holt-Winters and documented short-history fallbacks."""
    if not points:
        empty = pd.DataFrame(columns=["period", "revenue", "series"])
        return ForecastResult(empty, empty.copy(), "Unavailable - no historical revenue loaded")
    historical = pd.DataFrame(points)
    historical["period"] = pd.to_datetime(historical["period"], format="%Y-%m")
    historical = historical.sort_values("period", kind="mergesort").drop_duplicates(
        "period", keep="last"
    )
    complete_index = pd.date_range(
        historical["period"].min(), historical["period"].max(), freq="MS"
    )
    series = (
        historical.set_index("period")["revenue"]
        .astype("float64")
        .reindex(complete_index, fill_value=0.0)
    )
    if len(series) >= seasonal_periods * 2:
        fitted = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True, remove_bias=True)
        predicted = fitted.forecast(periods)
        method = f"Holt-Winters additive trend and {seasonal_periods}-month seasonality"
    elif len(series) >= 3:
        fitted = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True, remove_bias=True)
        predicted = fitted.forecast(periods)
        method = "Holt-Winters damped-trend fallback (insufficient seasonal history)"
    else:
        future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(), periods=periods)
        predicted = pd.Series([float(series.iloc[-1])] * periods, index=future_index)
        method = "Last-observation fallback (fewer than three historical months)"
    history_frame = series.rename("revenue").rename_axis("period").reset_index()
    history_frame["series"] = "Historical"
    forecast_frame = predicted.clip(lower=0.0).rename("revenue").rename_axis("period").reset_index()
    forecast_frame["series"] = "Forecast"
    return ForecastResult(history_frame, forecast_frame, method)
