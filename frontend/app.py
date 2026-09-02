"""Vantara Streamlit dashboard consuming only the governed FastAPI service."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

from frontend.analytics import friendly_feature_name, revenue_forecast
from frontend.api_client import APIClientError, DashboardAPI
from frontend.reports import batch_results_csv, batch_results_frame, batch_results_pdf
from frontend.styles import apply_dashboard_style, render_page_header
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "config.yaml")
DASHBOARD = CONFIG["dashboard"]
API_URL = os.getenv("VANTARA_API_URL", str(DASHBOARD["default_api_url"]))
TIMEOUT_SECONDS = float(DASHBOARD["request_timeout_seconds"])
CACHE_TTL_SECONDS = int(DASHBOARD["cache_ttl_seconds"])


@st.cache_resource
def api_client(base_url: str, timeout_seconds: float) -> DashboardAPI:
    """Create one shared HTTP client without loading models or database internals."""
    return DashboardAPI(base_url, timeout_seconds=timeout_seconds)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_overview(base_url: str, _api: DashboardAPI) -> dict[str, Any]:
    """Cache stable executive metrics for short dashboard rerun intervals."""
    del base_url
    return _api.overview()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_segments(base_url: str, _api: DashboardAPI) -> list[dict[str, Any]]:
    """Cache segment summaries for short dashboard rerun intervals."""
    del base_url
    return _api.segments()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_priority(
    base_url: str,
    segment_id: int | None,
    country: str | None,
    value_tier: str | None,
    limit: int,
    _api: DashboardAPI,
) -> list[dict[str, Any]]:
    """Cache the current filtered retention-priority list."""
    del base_url
    return _api.churn_priority(
        segment_id=segment_id,
        country=country,
        value_tier=value_tier,
        limit=limit,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_segment_customers(
    base_url: str,
    segment_id: int,
    country: str | None,
    value_tier: str | None,
    _api: DashboardAPI,
) -> list[dict[str, Any]]:
    """Cache one full filtered segment view."""
    del base_url
    return _api.segment_customers(
        segment_id,
        country=country,
        value_tier=value_tier,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_revenue(base_url: str, _api: DashboardAPI) -> list[dict[str, Any]]:
    """Cache monthly persisted revenue points."""
    del base_url
    return _api.revenue()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_metadata(base_url: str, _api: DashboardAPI) -> dict[str, Any]:
    """Cache safe frozen model metadata."""
    del base_url
    return _api.metadata()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_insights(base_url: str, _api: DashboardAPI) -> dict[str, Any]:
    """Cache global explainability and comparison evidence."""
    del base_url
    return _api.model_insights()


def _plotly_layout(figure: Figure, *, height: int = 390) -> Figure:
    figure.update_layout(
        height=height,
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#344649"},
        legend_title_text="",
    )
    return figure


def _selector_options(
    segments: list[dict[str, Any]], overview: dict[str, Any]
) -> tuple[dict[str, int], list[str], list[str]]:
    segment_lookup = {
        f"{row['segment_name']} ({row['customer_count']:,})": int(row["segment_id"])
        for row in segments
    }
    countries = ["All countries", *[str(value) for value in overview.get("countries", [])]]
    value_tiers = ["All value tiers", "high", "medium", "low", "unclassified"]
    return segment_lookup, countries, value_tiers


def _optional_filter(value: str, prefix: str) -> str | None:
    return None if value.startswith(prefix) else value


def _compact_currency(value: float) -> str:
    """Format large GBP values so KPI cards remain legible at laptop widths."""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"GBP {value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"GBP {value / 1_000:.1f}K"
    return f"GBP {value:,.0f}"


def render_executive_overview(api: DashboardAPI) -> None:
    """Render concise customer health, value, segments, and scored-priority context."""
    render_page_header(
        "Customer health, at a glance",
        "A decision-ready view of known customers, historical value, and the latest persisted "
        "retention signals.",
        eyebrow="Executive overview",
    )
    overview = cached_overview(API_URL, api)
    segments = cached_segments(API_URL, api)
    priorities = cached_priority(API_URL, None, None, None, 10, api)
    columns = st.columns(4)
    columns[0].metric("Known customers", f"{overview['total_customers']:,}")
    columns[1].metric("Customers scored", f"{overview['scored_customers']:,}")
    columns[2].metric(
        "Historical net value",
        _compact_currency(float(overview["total_historical_net_spend"])),
        help="Return-aware net spend observed before the governed feature cutoff.",
    )
    columns[3].metric("High-risk scored customers", f"{overview['high_risk_customers']:,}")
    left, right = st.columns([1.05, 1.45])
    with left:
        st.subheader("Customer mix")
        segment_frame = pd.DataFrame(segments)
        figure = px.bar(
            segment_frame.sort_values("customer_count"),
            x="customer_count",
            y="segment_name",
            orientation="h",
            color="customer_count",
            color_continuous_scale=["#cbd8d3", "#0f766e"],
            labels={"customer_count": "Customers", "segment_name": ""},
        )
        figure.update_coloraxes(showscale=False)
        st.plotly_chart(_plotly_layout(figure, height=420), use_container_width=True)
    with right:
        st.subheader("Current retention priorities")
        if priorities:
            frame = pd.DataFrame(priorities)
            display = frame.loc[
                :,
                [
                    "customer_id",
                    "segment_name",
                    "churn_probability",
                    "predicted_clv_180d",
                    "retention_priority",
                ],
            ].rename(
                columns={
                    "customer_id": "Customer",
                    "segment_name": "Segment",
                    "churn_probability": "Churn risk",
                    "predicted_clv_180d": "Predicted 180-day value",
                    "retention_priority": "Priority",
                }
            )
            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Churn risk": st.column_config.ProgressColumn(
                        format="percent", min_value=0.0, max_value=1.0
                    ),
                    "Predicted 180-day value": st.column_config.NumberColumn(format="GBP %.2f"),
                    "Priority": st.column_config.ProgressColumn(
                        format="%.3f", min_value=0.0, max_value=1.0
                    ),
                },
            )
        else:
            st.info(
                "No persisted scores are available yet. Score a known customer or upload a "
                "canonical batch to populate retention priorities."
            )
    st.caption(
        "Retention priority = normalized churn probability x normalized predicted 180-day value. "
        "Predictions support planning; they do not guarantee individual outcomes."
    )


def render_segments(api: DashboardAPI) -> None:
    """Render the required segment, country, and value-tier exploration surface."""
    render_page_header(
        "Understand every customer group",
        "Explore business-readable segments, filter the customer population, and compare the "
        "behavioral structure behind each group.",
        eyebrow="Customer segments",
    )
    overview = cached_overview(API_URL, api)
    segments = cached_segments(API_URL, api)
    insights = cached_insights(API_URL, api)
    segment_lookup, countries, value_tiers = _selector_options(segments, overview)
    filter_columns = st.columns([1.3, 1, 1])
    selected_label = filter_columns[0].selectbox("Segment", list(segment_lookup), key="segment")
    selected_country = filter_columns[1].selectbox("Country", countries, key="segment_country")
    selected_tier = filter_columns[2].selectbox("Value tier", value_tiers, key="segment_tier")
    segment_id = segment_lookup[selected_label]
    customers = cached_segment_customers(
        API_URL,
        segment_id,
        _optional_filter(selected_country, "All"),
        _optional_filter(selected_tier, "All"),
        api,
    )
    frame = pd.DataFrame(customers)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Customers in view", f"{len(frame):,}")
    metric_columns[1].metric(
        "Average historical value",
        "GBP 0.00" if frame.empty else f"GBP {frame['net_spend'].mean():,.2f}",
    )
    metric_columns[2].metric(
        "Countries represented",
        "0" if frame.empty else f"{frame['country'].nunique(dropna=True):,}",
    )
    left, right = st.columns([1.05, 1.35])
    with left:
        st.subheader("Segment profile")
        profiles = pd.DataFrame(insights["segment_profiles"])
        profile = profiles.loc[
            profiles["algorithm"].eq("kmeans")
            & profiles["kmeans_segment"].astype("Int64").eq(segment_id)
        ]
        if not profile.empty:
            row = profile.iloc[0]
            st.markdown(f"### {row['business_label']}")
            st.write(
                f"Typical recency is **{row['recency_days']:.0f} days**, with "
                f"**{row['frequency_orders']:.1f} orders** and average net spend of "
                f"**GBP {row['net_spend']:,.0f}**."
            )
        if frame.empty:
            st.info("No customers match all selected filters.")
        else:
            st.dataframe(
                frame.rename(
                    columns={
                        "customer_id": "Customer",
                        "country": "Country",
                        "net_spend": "Historical net spend",
                        "value_tier": "Value tier",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Historical net spend": st.column_config.NumberColumn(format="GBP %.2f")
                },
            )
    with right:
        st.subheader("Behavioral map")
        pca = pd.DataFrame(insights["segment_pca"])
        pca["kmeans_segment"] = pca["kmeans_segment"].astype(int)
        figure = px.scatter(
            pca,
            x="pca_1",
            y="pca_2",
            color=pca["kmeans_segment"].astype(str),
            opacity=0.55,
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"pca_1": "Behavior axis 1", "pca_2": "Behavior axis 2", "color": "Segment"},
        )
        st.plotly_chart(_plotly_layout(figure, height=520), use_container_width=True)
        st.caption(
            "PCA is used here only to visualize similarity. Segment assignment uses the governed "
            "behavioral features, not the two display axes."
        )


def render_churn_priority(api: DashboardAPI) -> None:
    """Render filtered high-value and high-risk customer prioritization."""
    render_page_header(
        "Focus retention effort where it matters",
        "Rank the latest persisted customer scores by both likelihood of churn and predicted "
        "180-day value.",
        eyebrow="Churn priority leaderboard",
    )
    overview = cached_overview(API_URL, api)
    segments = cached_segments(API_URL, api)
    segment_lookup, countries, value_tiers = _selector_options(segments, overview)
    filters = st.columns(3)
    segment_labels = ["All segments", *segment_lookup]
    selected_segment = filters[0].selectbox("Segment", segment_labels, key="risk_segment")
    selected_country = filters[1].selectbox("Country", countries, key="risk_country")
    selected_tier = filters[2].selectbox("Value tier", value_tiers, key="risk_tier")
    rows = cached_priority(
        API_URL,
        None if selected_segment == "All segments" else segment_lookup[selected_segment],
        _optional_filter(selected_country, "All"),
        _optional_filter(selected_tier, "All"),
        int(DASHBOARD["leaderboard_limit"]),
        api,
    )
    st.caption(
        "Priority uses min-max normalization within the current filtered population: normalized "
        "churn probability x normalized predicted 180-day value."
    )
    if not rows:
        st.info("No persisted predictions match these filters.")
        return
    frame = pd.DataFrame(rows)
    display = frame.loc[
        :,
        [
            "customer_id",
            "country",
            "value_tier",
            "segment_name",
            "churn_probability",
            "predicted_clv_180d",
            "retention_priority",
            "scored_at",
        ],
    ].rename(
        columns={
            "customer_id": "Customer",
            "country": "Country",
            "value_tier": "Value tier",
            "segment_name": "Segment",
            "churn_probability": "Churn risk",
            "predicted_clv_180d": "Predicted 180-day value",
            "retention_priority": "Priority",
            "scored_at": "Scored at",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Churn risk": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
            "Predicted 180-day value": st.column_config.NumberColumn(format="GBP %.2f"),
            "Priority": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "Scored at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    st.download_button(
        "Download leaderboard CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name="vantara_retention_priority.csv",
        mime="text/csv",
    )


def render_customer_explorer(api: DashboardAPI) -> None:
    """Render one-customer scoring, XAI, anomaly, and recommendation evidence."""
    render_page_header(
        "One customer, explained",
        "Score a known customer, see the prediction horizon and value, then review plain-language "
        "drivers and recommended products.",
        eyebrow="Customer explorer",
    )
    customer_id = st.text_input("Customer ID", placeholder="For example: 12346", key="customer_id")
    if st.button("Score customer", type="primary"):
        if not customer_id.strip():
            st.warning("Enter a customer ID before scoring.")
            return
        with st.spinner("Scoring customer and preparing explanation..."):
            try:
                identifier = customer_id.strip()
                st.session_state["customer_result"] = {
                    "score": api.score_customer(identifier),
                    "customer": api.customer(identifier),
                    "explanation": api.explanation(identifier),
                    "recommendations": api.recommendations(identifier),
                }
                cached_overview.clear()
                cached_priority.clear()
            except APIClientError as error:
                st.error(str(error))
                st.session_state.pop("customer_result", None)
    result = st.session_state.get("customer_result")
    if not result:
        st.info("Enter a known customer ID to generate a persisted, auditable score.")
        return
    score = result["score"]
    customer = result["customer"]
    explanation = result["explanation"]
    metrics = st.columns(4)
    metrics[0].metric("Churn risk", f"{score['churn_probability']:.1%}")
    metrics[1].metric("Predicted 180-day value", f"GBP {score['predicted_clv_180d']:,.2f}")
    metrics[2].metric(
        "Purchase in next 30 days",
        (
            "Not available"
            if score["next_purchase_probability"] is None
            else f"{score['next_purchase_probability']:.1%}"
        ),
    )
    metrics[3].metric("Customer segment", score["segment_name"])
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Why this churn score")
        explanation_text = str(explanation["text"])
        for driver in [
            *explanation["positive_drivers"],
            *explanation["negative_drivers"],
        ]:
            explanation_text = explanation_text.replace(
                str(driver), friendly_feature_name(str(driver))
            )
        st.write(explanation_text)
        driver_columns = st.columns(2)
        with driver_columns[0]:
            st.markdown("**Factors increasing risk**")
            for driver in explanation["positive_drivers"]:
                st.write(f"- {friendly_feature_name(driver)}")
        with driver_columns[1]:
            st.markdown("**Factors reducing risk**")
            for driver in explanation["negative_drivers"]:
                st.write(f"- {friendly_feature_name(driver)}")
        st.caption(
            "TreeSHAP attributes this model prediction; it does not establish that a factor caused "
            "the customer's behavior."
        )
    with right:
        st.subheader("Next best context")
        st.write(f"**Next category:** {score['next_category_id']}")
        st.write(f"**Country:** {customer.get('country') or 'Unknown'}")
        if score["anomaly_flag"]:
            st.warning(
                "This behavior is a manual-review anomaly candidate. It is not confirmed fraud."
            )
        else:
            st.success("No manual-review anomaly flag at the frozen threshold.")
        recommendations = pd.DataFrame(result["recommendations"])
        if not recommendations.empty:
            st.dataframe(
                recommendations[["rank", "stock_code"]].rename(
                    columns={"rank": "Rank", "stock_code": "Product code"}
                ),
                hide_index=True,
                use_container_width=True,
            )
    st.caption(
        f"Feature as-of: {score['as_of_timestamp']} | Model version: {score['model_version']} | "
        f"Persisted prediction ID: {score['prediction_id']}"
    )


def render_revenue(api: DashboardAPI) -> None:
    """Render historical revenue and the governed Holt-Winters forecast overlay."""
    render_page_header(
        "Revenue history, with a cautious look ahead",
        "Compare persisted monthly net merchandise revenue with a simple forecast. Forecasts are "
        "planning aids, not guarantees.",
        eyebrow="Revenue trends and forecast",
    )
    result = revenue_forecast(
        cached_revenue(API_URL, api),
        periods=int(DASHBOARD["forecast_periods"]),
        seasonal_periods=int(DASHBOARD["seasonal_periods"]),
    )
    if result.historical.empty:
        st.warning(
            "No transaction history is loaded in the serving database. Run the governed serving "
            "initializer with --include-transactions to enable revenue history and forecasting."
        )
        return
    combined = pd.concat([result.historical, result.forecast], ignore_index=True)
    figure = px.line(
        combined,
        x="period",
        y="revenue",
        color="series",
        markers=True,
        color_discrete_map={"Historical": "#183238", "Forecast": "#d9785f"},
        labels={"period": "Month", "revenue": "Net revenue (GBP)", "series": ""},
    )
    figure.update_traces(line={"width": 3})
    st.plotly_chart(_plotly_layout(figure, height=520), use_container_width=True)
    columns = st.columns(3)
    columns[0].metric("Historical months", f"{len(result.historical):,}")
    columns[1].metric("Forecast months", f"{len(result.forecast):,}")
    columns[2].metric(
        "Latest historical revenue", f"GBP {result.historical['revenue'].iloc[-1]:,.0f}"
    )
    st.info(f"Forecast method: {result.method}. Forecast values are estimates, not guarantees.")


def render_batch_scoring(api: DashboardAPI) -> None:
    """Render canonical CSV upload, scoring results, and audited downloads."""
    render_page_header(
        "Score a customer batch, safely",
        "Upload canonical transaction history. The API validates, derives frozen-schema features, "
        "persists each score, and returns auditable results.",
        eyebrow="Batch scoring",
    )
    template = (
        "invoice,stock_code,description,quantity,invoice_date,price,customer_id,country\n"
        "B001,85123A,WHITE HANGING HEART T-LIGHT HOLDER,2,"
        "2011-01-01T10:00:00,2.55,NEW-001,United Kingdom\n"
    )
    st.download_button(
        "Download canonical CSV template",
        template.encode("utf-8"),
        file_name="vantara_batch_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Canonical transaction CSV", type=["csv"], key="batch_file")
    st.caption(
        "Required columns, in order: invoice, stock_code, description, quantity, invoice_date, "
        "price, customer_id, country. Maximum 10 MB, 100,000 rows, and 1,000 customers."
    )
    if st.button("Validate and score batch", type="primary", disabled=uploaded is None):
        try:
            assert uploaded is not None
            with st.spinner("Validating transactions and scoring customers..."):
                st.session_state["batch_results"] = api.score_batch(
                    uploaded.name, uploaded.getvalue()
                )
                cached_overview.clear()
                cached_priority.clear()
        except APIClientError as error:
            st.error(str(error))
            st.session_state.pop("batch_results", None)
    rows = st.session_state.get("batch_results")
    if not rows:
        st.info("Validated batch results will appear here with CSV and PDF export options.")
        return
    frame = batch_results_frame(rows)
    metrics = st.columns(3)
    metrics[0].metric("Customers scored", f"{len(frame):,}")
    metrics[1].metric("Average churn risk", f"{frame['churn_probability'].mean():.1%}")
    metrics[2].metric("Predicted 180-day value", f"GBP {frame['predicted_clv_180d'].sum():,.2f}")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download complete CSV",
        batch_results_csv(rows),
        file_name="vantara_batch_scores.csv",
        mime="text/csv",
    )
    downloads[1].download_button(
        "Download audit PDF",
        batch_results_pdf(rows),
        file_name="vantara_batch_scores.pdf",
        mime="application/pdf",
    )
    st.caption(
        "Downloads retain customer identifiers, feature as-of timestamps, model version, scores, "
        "segments, persistence status, and manual-review flags."
    )


def render_model_insights(api: DashboardAPI) -> None:
    """Render global explainability, model comparisons, and honest held-out evidence."""
    render_page_header(
        "How the intelligence layer behaves",
        "Review the frozen model choice, global churn drivers, comparison evidence, and known "
        "limitations in business-readable terms.",
        eyebrow="Model insights and explainability",
    )
    metadata = cached_metadata(API_URL, api)
    insights = cached_insights(API_URL, api)
    columns = st.columns(3)
    columns[0].metric(
        "Production churn model", metadata["churn"]["model"].replace("_", " ").title()
    )
    columns[1].metric(
        "Held-out churn ROC-AUC", f"{metadata['churn']['held_out_metrics']['roc_auc']:.3f}"
    )
    columns[2].metric(
        "Held-out churn recall", f"{metadata['churn']['held_out_metrics']['recall']:.1%}"
    )
    left, right = st.columns([1.05, 1])
    importance = pd.DataFrame(insights["global_churn_importance"]).head(15)
    importance["Business feature"] = importance["feature"].map(friendly_feature_name)
    with left:
        st.subheader("Global churn drivers")
        figure = px.bar(
            importance.sort_values("mean_absolute_shap"),
            x="mean_absolute_shap",
            y="Business feature",
            orientation="h",
            color="mean_absolute_shap",
            color_continuous_scale=["#cbd8d3", "#0f766e"],
            labels={
                "mean_absolute_shap": "Average absolute prediction impact",
                "Business feature": "",
            },
        )
        figure.update_coloraxes(showscale=False)
        st.plotly_chart(_plotly_layout(figure, height=520), use_container_width=True)
    with right:
        st.subheader("Partial dependence")
        image_bytes = base64.b64decode(api.churn_partial_dependence())
        st.image(image_bytes, use_container_width=True)
        st.caption(
            "Partial dependence shows the model's average response while varying selected "
            "features; "
            "it is descriptive model behavior, not causal proof."
        )
    tabs = st.tabs(["Churn comparison", "Customer value", "Next category", "Recommendations"])
    tables = [
        ("churn_comparison", tabs[0]),
        ("clv_comparison", tabs[1]),
        ("next_category_evaluation", tabs[2]),
        ("recommender_evaluation", tabs[3]),
    ]
    for name, tab in tables:
        with tab:
            st.dataframe(pd.DataFrame(insights[name]), hide_index=True, use_container_width=True)
    clv_metrics = metadata["clv"]["held_out_metrics"]
    st.warning(
        f"Known limitation: held-out predicted-value R2 is {clv_metrics['r2']:.3f}, below the "
        "0.60 project target. The genuine result is preserved and must not be used as a guaranteed "
        "customer revenue estimate."
    )
    st.caption(
        f"Freeze: {metadata['freeze_version']} | "
        f"Feature schema: {metadata['feature_schema_version']} "
        f"({metadata['feature_count']} ordered inputs)"
    )


def main() -> None:
    """Run the Vantara Streamlit dashboard."""
    st.set_page_config(
        page_title="Vantara Customer Intelligence",
        page_icon="V",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_style()
    api = api_client(API_URL, TIMEOUT_SECONDS)
    with st.sidebar:
        st.markdown("## VANTARA")
        st.caption("Customer intelligence workspace")
        page = st.radio(
            "Workspace",
            [
                "Executive overview",
                "Customer segments",
                "Churn priorities",
                "Customer explorer",
                "Revenue and forecast",
                "Batch scoring",
                "Model insights",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        try:
            health = api.health()
            st.success(f"API {health['status']} | Database {health['database']}")
        except APIClientError:
            st.error("API unavailable")
        st.caption(f"Service: {API_URL}")
    renderers = {
        "Executive overview": render_executive_overview,
        "Customer segments": render_segments,
        "Churn priorities": render_churn_priority,
        "Customer explorer": render_customer_explorer,
        "Revenue and forecast": render_revenue,
        "Batch scoring": render_batch_scoring,
        "Model insights": render_model_insights,
    }
    try:
        renderers[page](api)
    except APIClientError as error:
        st.error(str(error))
        st.info(
            "Your selections and uploaded file remain unchanged. Retry after the API is healthy."
        )


if __name__ == "__main__":
    main()
