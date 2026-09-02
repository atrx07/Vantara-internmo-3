"""SQLAlchemy ORM entities for Vantara serving persistence."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by ORM entities and Alembic."""


class Customer(Base):
    """Opaque customer identity plus deterministic serving feature state."""

    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    country: Mapped[str | None] = mapped_column(String(128), index=True)
    feature_as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_payload: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    sequence_payload: Mapped[dict[str, object] | None] = mapped_column(JSON)
    net_spend: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False, default=0)
    value_tier: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    segment: Mapped[Segment | None] = relationship(
        back_populates="customer", cascade="all, delete-orphan", uselist=False
    )
    recommendations: Mapped[list[Recommendation]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )


class Transaction(Base):
    """Canonical transaction record used for serving analytics when loaded."""

    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_customer_timestamp", "customer_id", "invoice_timestamp"),
        Index("ix_transactions_invoice", "invoice"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice: Mapped[str] = mapped_column(String(64), nullable=False)
    stock_code: Mapped[str] = mapped_column(String(64), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="SET NULL"), nullable=True
    )
    description: Mapped[str | None] = mapped_column(String(512))
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    invoice_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    country: Mapped[str] = mapped_column(String(128), nullable=False)
    is_product: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_return: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_valid_merchandise: Mapped[bool] = mapped_column(Boolean, nullable=False)


class Prediction(Base):
    """Versioned multi-output score persisted for every API scoring action."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("churn_probability >= 0 AND churn_probability <= 1"),
        CheckConstraint(
            "next_purchase_probability IS NULL OR "
            "(next_purchase_probability >= 0 AND next_purchase_probability <= 1)"
        ),
        Index("ix_predictions_customer_scored", "customer_id", "scored_at"),
        Index("ix_predictions_risk_value", "churn_probability", "predicted_clv_180d"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    as_of_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    churn_probability: Mapped[Decimal] = mapped_column(Numeric(12, 10), nullable=False)
    churn_label: Mapped[bool] = mapped_column(Boolean, nullable=False)
    churn_threshold: Mapped[Decimal] = mapped_column(Numeric(12, 10), nullable=False)
    churn_threshold_version: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_clv_180d: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    next_purchase_probability: Mapped[Decimal | None] = mapped_column(Numeric(12, 10))
    next_category_id: Mapped[str | None] = mapped_column(String(64))
    next_category_probability: Mapped[Decimal | None] = mapped_column(Numeric(12, 10))
    anomaly_score: Mapped[Decimal] = mapped_column(Numeric(18, 10), nullable=False)
    anomaly_flag: Mapped[bool] = mapped_column(Boolean, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="predictions")


class Segment(Base):
    """Current business-readable customer segment assignment."""

    __tablename__ = "segments"
    __table_args__ = (Index("ix_segments_identifier_name", "segment_id", "segment_name"),)

    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="CASCADE"), primary_key=True
    )
    segment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="segment")


class Recommendation(Base):
    """Ranked deterministic product recommendation for a customer."""

    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint(
            "customer_id", "rank", "recommendation_version", name="uq_recommendation_rank"
        ),
        CheckConstraint("rank > 0"),
        Index("ix_recommendations_customer_rank", "customer_id", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    stock_code: Mapped[str] = mapped_column(String(64), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(14, 10), nullable=False)
    recommendation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="recommendations")
