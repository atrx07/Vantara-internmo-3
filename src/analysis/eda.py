"""Reusable STEP 03 EDA, evidence, visualization, and data-freeze logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.validation import DataValidationError
from src.features.schema import (
    FeatureSchemaEvidence,
    freeze_churn_feature_schema,
    write_feature_schema,
)
from src.utils.config import load_config, resolve_project_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EDAEvidence:
    """Required EDA tables and written modeling hypotheses."""

    rfm_summary: pd.DataFrame
    churn_rfm_comparison: pd.DataFrame
    class_balance: pd.DataFrame
    country_summary: pd.DataFrame
    seasonality_summary: pd.DataFrame
    outlier_summary: pd.DataFrame
    hypotheses: tuple[str, ...]


def build_eda_evidence(
    cleaned: pd.DataFrame,
    churn_features: pd.DataFrame,
    *,
    top_countries: int,
) -> EDAEvidence:
    """Compute all PRD-required EDA summaries without notebook-owned logic."""
    required_features = {"recency_days", "frequency_orders", "net_spend", "churn"}
    if not required_features.issubset(churn_features.columns):
        missing = sorted(required_features - set(churn_features.columns))
        raise DataValidationError(f"EDA churn table missing columns: {missing}")
    rfm_columns = [
        "recency_days",
        "frequency_orders",
        "gross_spend",
        "net_spend",
        "avg_order_value",
    ]
    rfm_summary = (
        churn_features[rfm_columns]
        .describe(percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        .transpose()
        .reset_index(names="feature")
    )
    churn_rfm_comparison = (
        churn_features.groupby("churn", observed=True)[rfm_columns]
        .agg(["median", "mean"])
        .reset_index()
    )
    churn_rfm_comparison.columns = [
        (
            "_".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
        )
        for column in churn_rfm_comparison.columns
    ]

    class_balance = (
        churn_features["churn"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("churn")
        .reset_index(name="customers")
    )
    class_balance["proportion"] = class_balance["customers"] / len(churn_features)

    positive = cleaned.loc[cleaned["is_positive_purchase"]].copy()
    country_summary = (
        positive.groupby("country", observed=True)
        .agg(
            orders=("invoice", "nunique"),
            identified_customers=("customer_id", "nunique"),
            gross_revenue=("gross_positive_value", "sum"),
            units=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("gross_revenue", ascending=False, kind="mergesort", ignore_index=True)
    )
    country_summary = country_summary.head(top_countries)

    positive["year_month"] = positive["invoice_date"].dt.to_period("M").astype(str)
    positive["quarter"] = positive["invoice_date"].dt.quarter.astype("int8")
    seasonality_summary = (
        positive.groupby(["year_month", "quarter"], observed=True)
        .agg(
            orders=("invoice", "nunique"),
            gross_revenue=("gross_positive_value", "sum"),
            units=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("year_month", kind="mergesort", ignore_index=True)
    )

    audit_flags = [
        "is_missing_customer_id",
        "is_return",
        "is_cancelled_invoice",
        "is_non_positive_price",
        "is_administrative_line",
        "is_statistical_outlier",
        "is_likely_data_error",
    ]
    outlier_summary = pd.DataFrame(
        {
            "quality_flag": audit_flags,
            "rows": [int(cleaned[column].sum()) for column in audit_flags],
        }
    )
    outlier_summary["proportion"] = outlier_summary["rows"] / len(cleaned)

    comparison = churn_features.groupby("churn", observed=True).median(numeric_only=True)
    active = comparison.loc[0]
    churned = comparison.loc[1]
    leading_country = str(country_summary.iloc[0]["country"])
    leading_share = float(
        country_summary.iloc[0]["gross_revenue"] / country_summary["gross_revenue"].sum()
    )
    peak_month = str(
        seasonality_summary.loc[seasonality_summary["gross_revenue"].idxmax(), "year_month"]
    )
    hypotheses = (
        f"H1: Higher recency should increase churn risk; training evidence shows median recency "
        f"{churned['recency_days']:.1f} days for churned versus "
        f"{active['recency_days']:.1f} for active customers.",
        f"H2: Higher order frequency should reduce churn risk; medians are "
        f"{churned['frequency_orders']:.1f} versus {active['frequency_orders']:.1f} orders.",
        "H3: Net spend and basket behavior should add value beyond frequency, while "
        "correlation/VIF evidence must prevent redundant monetary inputs.",
        "H4: Return rate and markdown affinity may identify behaviorally distinct risk groups "
        "and should be tested as interactions rather than causal effects.",
        f"H5: Seasonality should affect purchase timing; observed gross revenue peaks in "
        f"{peak_month} and must be interpreted using pre-cutoff features only.",
        f"H6: Country mix may affect behavior; {leading_country} contributes "
        f"{leading_share:.1%} of revenue among the displayed top countries, so country "
        "results require scale-aware interpretation.",
        "H7: Frozen category affinities and training-only product popularity should improve "
        "behavioral separation without target leakage.",
    )
    return EDAEvidence(
        rfm_summary=rfm_summary,
        churn_rfm_comparison=churn_rfm_comparison,
        class_balance=class_balance,
        country_summary=country_summary,
        seasonality_summary=seasonality_summary,
        outlier_summary=outlier_summary,
        hypotheses=hypotheses,
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "Vantara"})
    plt.close(figure)


def create_eda_figures(
    cleaned: pd.DataFrame,
    churn_features: pd.DataFrame,
    evidence: EDAEvidence,
    schema: FeatureSchemaEvidence,
    output_directory: Path,
) -> None:
    """Create deterministic static figures for notebook and report consumption."""
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(churn_features["recency_days"], bins=40, color="#356859")
    axes[0].set_title("Recency (days)")
    axes[1].hist(np.log1p(churn_features["frequency_orders"]), bins=40, color="#6B8E6E")
    axes[1].set_title("log1p Order frequency")
    axes[2].hist(np.log1p(churn_features["net_spend"]), bins=40, color="#C07A54")
    axes[2].set_title("log1p Net spend")
    figure.suptitle("Customer RFM distributions")
    _save_figure(figure, output_directory / "rfm_distributions.png")

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        evidence.class_balance["churn"].astype(str),
        evidence.class_balance["customers"],
        color=["#356859", "#C07A54"],
    )
    axis.set(title="Canonical churn class balance", xlabel="Churn", ylabel="Customers")
    _save_figure(figure, output_directory / "churn_class_balance.png")

    figure, axis = plt.subplots(figsize=(10, 5))
    countries = evidence.country_summary.sort_values("gross_revenue")
    axis.barh(countries["country"], countries["gross_revenue"], color="#567D8C")
    axis.set(title="Top-country gross merchandise revenue", xlabel="Revenue")
    _save_figure(figure, output_directory / "country_revenue.png")

    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(
        evidence.seasonality_summary["year_month"],
        evidence.seasonality_summary["orders"],
        color="#8A5A44",
        marker="o",
        linewidth=1.5,
    )
    axis.tick_params(axis="x", rotation=70)
    axis.set(title="Monthly valid positive orders", xlabel="Month", ylabel="Orders")
    _save_figure(figure, output_directory / "monthly_orders.png")

    core = list(schema.selected_features[: min(20, len(schema.selected_features))])
    figure, axis = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        churn_features.loc[churn_features["partition"] == "train", core].corr(),
        cmap="vlag",
        center=0,
        square=False,
        ax=axis,
    )
    axis.set_title("Training-only selected-feature correlation")
    _save_figure(figure, output_directory / "selected_feature_correlation.png")

    del cleaned


def _write_hypotheses(hypotheses: tuple[str, ...], path: Path) -> None:
    lines = ["# STEP 03 Modeling Hypotheses", ""]
    for hypothesis in hypotheses:
        lines.extend([f"- {hypothesis}", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_step03_analysis(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate required EDA evidence and freeze the train-only churn schema."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    cleaned_path = resolve_project_path(
        config["cleaning"]["interim_transactions"], project_root=root
    )
    churn_path = resolve_project_path(config["outputs"]["churn_features"], project_root=root)
    if not cleaned_path.is_file() or not churn_path.is_file():
        raise DataValidationError("STEP 02 outputs are required before STEP 03 analysis")
    cleaned = pd.read_parquet(cleaned_path)
    churn_features = pd.read_parquet(churn_path)
    analysis_config = config["analysis"]
    evidence_directory = resolve_project_path(
        analysis_config["evidence_directory"], project_root=root
    )
    freeze_directory = resolve_project_path(
        analysis_config["data_freeze_directory"], project_root=root
    )
    schema_path = resolve_project_path(analysis_config["churn_feature_schema"], project_root=root)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    freeze_directory.mkdir(parents=True, exist_ok=True)

    evidence = build_eda_evidence(
        cleaned, churn_features, top_countries=int(analysis_config["top_countries"])
    )
    schema = freeze_churn_feature_schema(
        churn_features,
        schema_version=str(analysis_config["feature_schema_version"]),
        correlation_threshold=float(analysis_config["correlation_threshold"]),
        vif_threshold=float(analysis_config["vif_threshold"]),
    )
    tables = {
        "rfm_summary.csv": evidence.rfm_summary,
        "churn_rfm_comparison.csv": evidence.churn_rfm_comparison,
        "class_balance.csv": evidence.class_balance,
        "country_summary.csv": evidence.country_summary,
        "seasonality_summary.csv": evidence.seasonality_summary,
        "outlier_summary.csv": evidence.outlier_summary,
    }
    for filename, table in tables.items():
        table.to_csv(evidence_directory / filename, index=False, float_format="%.10g")
    _write_hypotheses(evidence.hypotheses, evidence_directory / "hypotheses.md")
    schema.correlation_matrix.to_csv(freeze_directory / "correlation_matrix.csv")
    schema.high_correlation_pairs.to_csv(
        freeze_directory / "high_correlation_pairs.csv", index=False, float_format="%.10g"
    )
    schema.initial_vif.to_csv(
        freeze_directory / "vif_initial.csv", index=False, float_format="%.10g"
    )
    schema.final_vif.to_csv(freeze_directory / "vif_final.csv", index=False, float_format="%.10g")
    write_feature_schema(schema, schema_path)
    create_eda_figures(cleaned, churn_features, evidence, schema, evidence_directory)

    summary: dict[str, Any] = {
        "step": "STEP 03",
        "schema_version": schema.schema_version,
        "selected_feature_count": len(schema.selected_features),
        "excluded_feature_count": len(schema.exclusions),
        "selected_features": list(schema.selected_features),
        "exclusions": schema.exclusions,
        "maximum_final_vif": float(schema.final_vif["vif"].max()),
        "high_correlation_pair_count": len(schema.high_correlation_pairs),
        "cleaned_rows": len(cleaned),
        "customer_rows": len(churn_features),
        "churn_rate": float(churn_features["churn"].mean()),
        "hypothesis_count": len(evidence.hypotheses),
    }
    (freeze_directory / "data_freeze_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info(
        "STEP 03 EDA and data freeze passed",
        extra={"event": "step03_analysis_passed", "rows": len(churn_features)},
    )
    return summary
