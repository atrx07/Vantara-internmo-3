"""Deterministic serving-data initialization from governed processed artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, insert, select
from sqlalchemy.orm import Session

from api.artifacts import ArtifactRegistry
from api.batch import build_latest_sequence_payloads
from api.models import Customer, Prediction, Recommendation, Segment, Transaction


def _value_tiers(values: pd.Series) -> pd.Series:
    lower, upper = values.quantile([1 / 3, 2 / 3]).tolist()
    return pd.Series(
        np.select(
            [values.le(lower), values.le(upper)],
            ["low", "medium"],
            default="high",
        ),
        index=values.index,
        dtype="string",
    )


def _recommendations(
    transactions: pd.DataFrame,
    customer_ids: set[str],
    segment_lookup: dict[str, int],
    registry: ArtifactRegistry,
) -> list[Recommendation]:
    artifact = registry.recommender
    items = [str(value) for value in artifact["items"]]
    item_index = {value: index for index, value in enumerate(items)}
    histories = (
        transactions.loc[
            transactions["customer_id"].isin(customer_ids)
            & transactions["is_positive_purchase"].fillna(False)
            & transactions["is_product"].fillna(False),
            ["customer_id", "stock_code"],
        ]
        .drop_duplicates()
        .groupby("customer_id", observed=True)["stock_code"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    generated_at = datetime.now(UTC)
    rows: list[Recommendation] = []
    top_k = int(artifact["top_k"])
    global_popular = [str(value) for value in artifact["global_popular"]]
    segment_popular = {
        int(key): [str(value) for value in values]
        for key, values in artifact["segment_popular"].items()
    }
    for customer_id in sorted(customer_ids):
        seen_codes = histories.get(customer_id, set())
        seen = {item_index[value] for value in seen_codes if value in item_index}
        scores: dict[int, float] = {}
        for index in seen:
            for neighbor, similarity in zip(
                artifact["neighbor_indices"][index],
                artifact["similarities"][index],
                strict=True,
            ):
                candidate = int(neighbor)
                if candidate not in seen and float(similarity) > 0:
                    scores[candidate] = scores.get(candidate, 0.0) + float(similarity)
        ranked = [
            (items[index], score)
            for index, score in sorted(scores.items(), key=lambda pair: (-pair[1], items[pair[0]]))
        ]
        fallback = segment_popular.get(segment_lookup.get(customer_id, -1), global_popular)
        selected = ranked[:top_k]
        selected_codes = {stock_code for stock_code, _ in selected}
        if len(selected) < top_k:
            for stock_code in fallback:
                if stock_code not in seen_codes and stock_code not in selected_codes:
                    selected.append((stock_code, 0.0))
                    selected_codes.add(stock_code)
                if len(selected) >= top_k:
                    break
        rows.extend(
            Recommendation(
                customer_id=customer_id,
                stock_code=stock_code,
                rank=rank,
                score=score,
                recommendation_version="item-cosine-production-v1",
                generated_at=generated_at,
            )
            for rank, (stock_code, score) in enumerate(selected, start=1)
        )
    return rows


def _transaction_records(
    transactions: pd.DataFrame, customer_ids: set[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in transactions.itertuples(index=False):
        customer_id = None if pd.isna(row.customer_id) else str(row.customer_id)
        records.append(
            {
                "invoice": str(row.invoice),
                "stock_code": str(row.stock_code),
                "customer_id": customer_id if customer_id in customer_ids else None,
                "description": None if pd.isna(row.description) else str(row.description),
                "quantity": int(row.quantity),
                "price": float(row.price),
                "invoice_timestamp": pd.Timestamp(row.invoice_date).to_pydatetime(),
                "country": str(row.country),
                "is_product": bool(row.is_product),
                "is_return": bool(row.is_return),
                "is_valid_merchandise": bool(row.is_valid_merchandise),
            }
        )
    return records


def initialize_serving_data(
    engine: Engine,
    project_root: Path,
    registry: ArtifactRegistry,
    config: dict[str, Any],
    *,
    include_transactions: bool = False,
    replace: bool = False,
) -> dict[str, int]:
    """Load customer features, sequences, segments, recommendations, and optional transactions."""
    with Session(engine) as session:
        existing = int(session.scalar(select(func.count()).select_from(Customer)) or 0)
        if existing and not replace:
            return {
                "customers": existing,
                "segments": int(session.scalar(select(func.count()).select_from(Segment)) or 0),
                "recommendations": int(
                    session.scalar(select(func.count()).select_from(Recommendation)) or 0
                ),
                "transactions": int(
                    session.scalar(select(func.count()).select_from(Transaction)) or 0
                ),
            }
        if replace:
            for entity in (Recommendation, Prediction, Segment, Transaction, Customer):
                session.execute(delete(entity))
            session.commit()

    churn = pd.read_parquet(project_root / config["outputs"]["churn_features"])
    transactions = pd.read_parquet(project_root / config["cleaning"]["interim_transactions"])
    taxonomy = pd.read_parquet(
        registry.artifact_root / "product_taxonomy" / "product_taxonomy.parquet"
    )
    cutoff_values = churn["cutoff_timestamp"].drop_duplicates()
    if len(cutoff_values) != 1:
        raise RuntimeError("Serving customer features must share one cutoff")
    cutoff = pd.Timestamp(cutoff_values.iloc[0])
    customer_ids = set(churn["customer_id"].astype(str))
    sequences = build_latest_sequence_payloads(
        transactions,
        taxonomy,
        cutoff=cutoff,
        sequence_length=int(config["deep_learning"]["lstm"]["sequence_length"]),
        minimum_events=int(config["deep_learning"]["lstm"]["minimum_history_events"]),
    )
    latest_country = (
        transactions.loc[transactions["customer_id"].isin(customer_ids)]
        .sort_values("invoice_date", kind="mergesort")
        .drop_duplicates("customer_id", keep="last")
        .set_index("customer_id")["country"]
        .astype(str)
        .to_dict()
    )
    churn = churn.copy()
    churn["value_tier"] = _value_tiers(churn["net_spend"].astype("float64"))
    segment_features = list(registry.segmentation["feature_names"])
    scaled = registry.segmentation["preprocessor"].transform(churn.loc[:, segment_features])
    segment_ids = registry.segmentation["kmeans"].predict(scaled).astype(int)
    segment_lookup = dict(zip(churn["customer_id"].astype(str), segment_ids, strict=True))
    loaded_at = datetime.now(UTC)
    customers: list[Customer] = []
    segments: list[Segment] = []
    for (_, row), segment_id in zip(churn.iterrows(), segment_ids, strict=True):
        customer_id = str(row["customer_id"])
        customers.append(
            Customer(
                customer_id=customer_id,
                country=latest_country.get(customer_id),
                feature_as_of=cutoff.to_pydatetime(),
                feature_schema_version=str(registry.freeze["feature_schema_version"]),
                feature_payload={name: float(row[name]) for name in registry.feature_names},
                sequence_payload=sequences.get(customer_id),
                net_spend=float(row["net_spend"]),
                value_tier=str(row["value_tier"]),
                source_sha256=str(registry.freeze["source_sha256"]),
                loaded_at=loaded_at,
            )
        )
        segments.append(
            Segment(
                customer_id=customer_id,
                segment_id=int(segment_id),
                segment_name=str(
                    registry.segmentation["kmeans_labels"].get(
                        int(segment_id), f"Segment {segment_id}"
                    )
                ),
                model_version="kmeans-production-v1",
                assigned_at=loaded_at,
            )
        )
    recommendations = _recommendations(transactions, customer_ids, segment_lookup, registry)
    with Session(engine) as session:
        session.add_all(customers)
        session.flush()
        session.add_all(segments)
        session.add_all(recommendations)
        session.commit()
        if include_transactions:
            for start in range(0, len(transactions), 5000):
                records = _transaction_records(
                    transactions.iloc[start : start + 5000], customer_ids
                )
                session.execute(insert(Transaction), records)
                session.commit()
        return {
            "customers": len(customers),
            "segments": len(segments),
            "recommendations": len(recommendations),
            "transactions": len(transactions) if include_transactions else 0,
        }
