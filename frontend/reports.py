"""Auditable CSV and PDF exports for dashboard batch-scoring results."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BRAND_NAVY = colors.HexColor("#183238")
BRAND_TEAL = colors.HexColor("#0F766E")
BRAND_SAND = colors.HexColor("#E7E3DA")
BRAND_INK = colors.HexColor("#182528")
BRAND_MUTED = colors.HexColor("#5D6B6E")


def batch_results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a stable, audit-friendly batch result table."""
    columns = [
        "prediction_id",
        "customer_id",
        "as_of_timestamp",
        "churn_probability",
        "churn_label",
        "churn_threshold",
        "predicted_clv_180d",
        "next_purchase_probability",
        "next_category_id",
        "next_category_probability",
        "anomaly_score",
        "anomaly_flag",
        "segment_id",
        "segment_name",
        "model_version",
        "persisted",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame.loc[:, columns]


def batch_results_csv(rows: list[dict[str, Any]]) -> bytes:
    """Serialize complete batch scores as UTF-8 CSV bytes."""
    return batch_results_frame(rows).to_csv(index=False).encode("utf-8")


def _page_decorator(generated_at: datetime) -> Callable[[Canvas, BaseDocTemplate], None]:
    def decorate(canvas: Canvas, document: BaseDocTemplate) -> None:
        canvas.saveState()
        width, _ = landscape(A4)
        canvas.setStrokeColor(BRAND_SAND)
        canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        canvas.setFillColor(BRAND_MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            18 * mm,
            8 * mm,
            f"Vantara batch scoring report | Generated {generated_at.isoformat()}",
        )
        page_text = f"Page {document.page}"
        canvas.drawString(
            width - 18 * mm - stringWidth(page_text, "Helvetica", 7.5),
            8 * mm,
            page_text,
        )
        canvas.restoreState()

    return decorate


def batch_results_pdf(
    rows: list[dict[str, Any]],
    *,
    generated_at: datetime | None = None,
) -> bytes:
    """Render a paginated landscape PDF with scoring and audit metadata."""
    if not rows:
        raise ValueError("At least one batch score is required for a PDF report")
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    frame = batch_results_frame(rows)
    model_versions = sorted(frame["model_version"].dropna().astype(str).unique())
    as_of_values = sorted(frame["as_of_timestamp"].dropna().astype(str).unique())
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Vantara Batch Scoring Report",
        author="Vantara Retail Solutions",
        subject="Auditable customer behavior prediction results",
        pageCompression=0,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="VantaraTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=25,
            textColor=BRAND_NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VantaraBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=BRAND_INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VantaraSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            textColor=BRAND_MUTED,
        )
    )
    story: list[Any] = [
        Paragraph("Vantara Customer Intelligence", styles["VantaraTitle"]),
        Paragraph(
            "Batch scoring report - predicted values support retention planning and are not "
            "guarantees of future behavior.",
            styles["VantaraBody"],
        ),
        Spacer(1, 5 * mm),
    ]
    metadata = [
        ["Customers scored", f"{len(frame):,}"],
        ["Report generated", timestamp.isoformat()],
        ["Model version", ", ".join(model_versions) or "Not provided"],
        ["Feature as-of", ", ".join(as_of_values[:3]) or "Not provided"],
    ]
    metadata_table = Table(metadata, colWidths=[38 * mm, 190 * mm])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), BRAND_SAND),
                ("TEXTCOLOR", (0, 0), (0, -1), BRAND_NAVY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    average_risk = float(frame["churn_probability"].astype(float).mean())
    projected_value = float(frame["predicted_clv_180d"].astype(float).sum())
    review_count = int(frame["anomaly_flag"].fillna(False).astype(bool).sum())
    summary = Table(
        [
            ["Average churn risk", "Predicted 180-day value", "Manual-review candidates"],
            [f"{average_risk:.1%}", f"GBP {projected_value:,.2f}", f"{review_count:,}"],
        ],
        colWidths=[76 * mm, 76 * mm, 76 * mm],
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F8F7F3")),
                ("TEXTCOLOR", (0, 1), (-1, 1), BRAND_TEAL),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, 1), 13),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (-1, -1), 0.5, BRAND_SAND),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, BRAND_SAND),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([metadata_table, Spacer(1, 5 * mm), summary, Spacer(1, 7 * mm)])
    headers = [
        "Customer",
        "Churn risk",
        "Predicted value",
        "Purchase in 30d",
        "Next category",
        "Segment",
        "Review flag",
        "Persisted",
    ]
    table_rows: list[list[Any]] = [headers]
    for row in frame.to_dict(orient="records"):
        next_probability = row["next_purchase_probability"]
        table_rows.append(
            [
                Paragraph(str(row["customer_id"]), styles["VantaraSmall"]),
                f"{float(row['churn_probability']):.1%}",
                f"GBP {float(row['predicted_clv_180d']):,.2f}",
                "N/A" if pd.isna(next_probability) else f"{float(next_probability):.1%}",
                Paragraph(str(row["next_category_id"]), styles["VantaraSmall"]),
                Paragraph(str(row["segment_name"]), styles["VantaraSmall"]),
                "Review" if bool(row["anomaly_flag"]) else "No flag",
                "Yes" if bool(row["persisted"]) else "No",
            ]
        )
    results_table = Table(
        table_rows,
        repeatRows=1,
        colWidths=[25 * mm, 25 * mm, 31 * mm, 31 * mm, 32 * mm, 40 * mm, 24 * mm, 20 * mm],
    )
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("TEXTCOLOR", (0, 1), (-1, -1), BRAND_INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3EE")]),
                ("GRID", (0, 0), (-1, -1), 0.25, BRAND_SAND),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    note = KeepTogether(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "Interpretation: churn risk is a predicted probability. Predicted value is the "
                "180-day forward net-revenue proxy. Anomaly flags indicate manual-review "
                "candidates and are not confirmed fraud.",
                styles["VantaraSmall"],
            ),
        ]
    )
    story.extend([results_table, note])
    decorator = _page_decorator(timestamp)
    document.build(story, onFirstPage=decorator, onLaterPages=decorator)
    return buffer.getvalue()
