"""Create Vantara PostgreSQL serving schema.

Revision ID: 20260901_0001
Revises: None
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260901_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create customers, transactions, predictions, segments, and recommendations."""
    op.create_table(
        "customers",
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("feature_as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column("feature_payload", sa.JSON(), nullable=False),
        sa.Column("sequence_payload", sa.JSON(), nullable=True),
        sa.Column("net_spend", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("value_tier", sa.String(length=32), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("customer_id"),
    )
    op.create_index("ix_customers_country", "customers", ["country"])
    op.create_index("ix_customers_value_tier", "customers", ["value_tier"])

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("invoice", sa.String(length=64), nullable=False),
        sa.Column("stock_code", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=True),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("invoice_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("country", sa.String(length=128), nullable=False),
        sa.Column("is_product", sa.Boolean(), nullable=False),
        sa.Column("is_return", sa.Boolean(), nullable=False),
        sa.Column("is_valid_merchandise", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_transactions_customer_timestamp", "transactions", ["customer_id", "invoice_timestamp"]
    )
    op.create_index("ix_transactions_invoice", "transactions", ["invoice"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("churn_probability", sa.Numeric(precision=12, scale=10), nullable=False),
        sa.Column("churn_label", sa.Boolean(), nullable=False),
        sa.Column("churn_threshold", sa.Numeric(precision=12, scale=10), nullable=False),
        sa.Column("churn_threshold_version", sa.String(length=64), nullable=False),
        sa.Column("predicted_clv_180d", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("next_purchase_probability", sa.Numeric(precision=12, scale=10), nullable=True),
        sa.Column("next_category_id", sa.String(length=64), nullable=True),
        sa.Column("next_category_probability", sa.Numeric(precision=12, scale=10), nullable=True),
        sa.Column("anomaly_score", sa.Numeric(precision=18, scale=10), nullable=False),
        sa.Column("anomaly_flag", sa.Boolean(), nullable=False),
        sa.CheckConstraint("churn_probability >= 0 AND churn_probability <= 1"),
        sa.CheckConstraint(
            "next_purchase_probability IS NULL OR "
            "(next_purchase_probability >= 0 AND next_purchase_probability <= 1)"
        ),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_customer_scored", "predictions", ["customer_id", "scored_at"])
    op.create_index(
        "ix_predictions_risk_value", "predictions", ["churn_probability", "predicted_clv_180d"]
    )

    op.create_table(
        "segments",
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("segment_id", sa.Integer(), nullable=False),
        sa.Column("segment_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("customer_id"),
    )
    op.create_index("ix_segments_identifier_name", "segments", ["segment_id", "segment_name"])

    op.create_table(
        "recommendations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("stock_code", sa.String(length=64), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Numeric(precision=14, scale=10), nullable=False),
        sa.Column("recommendation_version", sa.String(length=64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("rank > 0"),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.customer_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "customer_id", "rank", "recommendation_version", name="uq_recommendation_rank"
        ),
    )
    op.create_index("ix_recommendations_customer_rank", "recommendations", ["customer_id", "rank"])


def downgrade() -> None:
    """Drop the serving schema in reverse dependency order."""
    op.drop_index("ix_recommendations_customer_rank", table_name="recommendations")
    op.drop_table("recommendations")
    op.drop_index("ix_segments_identifier_name", table_name="segments")
    op.drop_table("segments")
    op.drop_index("ix_predictions_risk_value", table_name="predictions")
    op.drop_index("ix_predictions_customer_scored", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_transactions_invoice", table_name="transactions")
    op.drop_index("ix_transactions_customer_timestamp", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index("ix_customers_value_tier", table_name="customers")
    op.drop_index("ix_customers_country", table_name="customers")
    op.drop_table("customers")
