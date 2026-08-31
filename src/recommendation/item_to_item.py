"""Training-customer item-to-item implicit recommender with offline evaluation."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from src.data.validation import DataValidationError
from src.models.common import log_mlflow_run, save_joblib_artifact


def _positive_product_lines(transactions: pd.DataFrame, customer_ids: set[str]) -> pd.DataFrame:
    eligible = transactions.loc[
        transactions["customer_id"].isin(customer_ids)
        & transactions["is_positive_purchase"].fillna(False)
        & transactions["is_product"].fillna(False),
        ["customer_id", "invoice", "invoice_date", "stock_code", "quantity"],
    ].copy()
    if eligible.empty:
        raise DataValidationError("No training-customer product interactions are available")
    return eligible


def _leave_last_order_out(lines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = (
        lines[["customer_id", "invoice", "invoice_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "invoice_date", "invoice"], kind="mergesort")
    )
    eligible_customers = orders.groupby("customer_id", observed=True).size()
    eligible_customers = set(eligible_customers[eligible_customers.ge(2)].index.astype("string"))
    evaluation_lines = lines.loc[lines["customer_id"].isin(eligible_customers)].copy()
    last_orders = orders.loc[orders["customer_id"].isin(eligible_customers)].drop_duplicates(
        "customer_id", keep="last"
    )
    marked = evaluation_lines.merge(
        last_orders[["customer_id", "invoice"]].assign(is_holdout=True),
        on=["customer_id", "invoice"],
        how="left",
        validate="many_to_one",
    )
    is_holdout = marked["is_holdout"].eq(True)
    holdout = marked.loc[is_holdout].drop(columns="is_holdout")
    history = marked.loc[~is_holdout].drop(columns="is_holdout")
    return history, holdout


def _interaction_matrix(
    lines: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[str], list[str], dict[str, int], dict[str, int]]:
    interactions = (
        lines.groupby(["customer_id", "stock_code"], observed=True, as_index=False)["quantity"]
        .sum()
        .loc[lambda frame: frame["quantity"].gt(0)]
    )
    customers = sorted(interactions["customer_id"].astype("string").unique())
    items = sorted(interactions["stock_code"].astype("string").unique())
    customer_index = {value: index for index, value in enumerate(customers)}
    item_index = {value: index for index, value in enumerate(items)}
    rows = interactions["customer_id"].astype("string").map(customer_index).to_numpy()
    columns = interactions["stock_code"].astype("string").map(item_index).to_numpy()
    values = np.log1p(interactions["quantity"].astype("float64").to_numpy())
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(customers), len(items)))
    return matrix, customers, items, customer_index, item_index


def _neighbor_arrays(
    matrix: sparse.csr_matrix,
    *,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    item_vectors = matrix.T.tocsr()
    count = min(neighbors + 1, item_vectors.shape[0])
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=count, n_jobs=1)
    model.fit(item_vectors)
    distances, indices = model.kneighbors(item_vectors)
    return indices[:, 1:].astype("int32"), (1.0 - distances[:, 1:]).astype("float32")


def _popular_by_segment(
    lines: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    limit: int,
) -> dict[int, list[str]]:
    merged = lines.merge(
        assignments[["customer_id", "kmeans_segment"]],
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )
    popularity = (
        merged.groupby(["kmeans_segment", "stock_code"], observed=True, as_index=False)["quantity"]
        .sum()
        .sort_values(
            ["kmeans_segment", "quantity", "stock_code"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    return {
        int(segment): list(group.head(limit)["stock_code"].astype("string"))
        for segment, group in popularity.groupby("kmeans_segment", observed=True)
    }


def _recommend(
    seen_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    *,
    items: Sequence[str],
    fallback: Sequence[str],
    top_k: int,
) -> list[str]:
    seen = set(int(value) for value in seen_indices)
    scores: defaultdict[int, float] = defaultdict(float)
    for item_index in seen:
        for neighbor, similarity in zip(
            neighbor_indices[item_index], similarities[item_index], strict=True
        ):
            candidate = int(neighbor)
            if candidate not in seen and float(similarity) > 0.0:
                scores[candidate] += float(similarity)
    ranked = [
        str(items[index])
        for index, _ in sorted(scores.items(), key=lambda pair: (-pair[1], str(items[pair[0]])))
    ]
    recommendations = ranked[:top_k]
    for stock_code in fallback:
        if stock_code not in recommendations and stock_code not in {items[index] for index in seen}:
            recommendations.append(str(stock_code))
        if len(recommendations) == top_k:
            break
    return recommendations


def _evaluate(
    history: pd.DataFrame,
    holdout: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    neighbors: int,
    top_k: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    matrix, customers, items, customer_index, item_index = _interaction_matrix(history)
    neighbor_indices, similarities = _neighbor_arrays(matrix, neighbors=neighbors)
    segment_popular = _popular_by_segment(history, assignments, limit=max(50, top_k))
    global_popular = list(
        history.groupby("stock_code", observed=True)["quantity"]
        .sum()
        .sort_values(ascending=False)
        .index.astype("string")[: max(50, top_k)]
    )
    segment_lookup = assignments.set_index("customer_id")["kmeans_segment"].to_dict()
    heldout_items = holdout.groupby("customer_id", observed=True)["stock_code"].agg(
        lambda values: set(values.astype("string"))
    )
    recall_values: list[float] = []
    hit_values: list[float] = []
    recommended_catalog: set[str] = set()
    evaluated = 0
    for customer_id, truth in heldout_items.items():
        customer = str(customer_id)
        if customer not in customer_index:
            continue
        row = matrix.getrow(customer_index[customer])
        segment = int(segment_lookup.get(customer, -1))
        fallback = segment_popular.get(segment, global_popular)
        recommendations = _recommend(
            row.indices,
            neighbor_indices,
            similarities,
            items=items,
            fallback=fallback,
            top_k=top_k,
        )
        hits = len(set(recommendations).intersection(truth))
        recall_values.append(hits / max(len(truth), 1))
        hit_values.append(float(hits > 0))
        recommended_catalog.update(recommendations)
        evaluated += 1
    if not evaluated:
        raise DataValidationError("No customers were eligible for recommender evaluation")
    metrics = {
        f"recall_at_{top_k}": float(np.mean(recall_values)),
        f"hit_rate_at_{top_k}": float(np.mean(hit_values)),
        "catalog_coverage": float(len(recommended_catalog) / max(len(items), 1)),
        "evaluated_customers": float(evaluated),
    }
    payload = {
        "items": items,
        "neighbor_indices": neighbor_indices,
        "similarities": similarities,
        "segment_popular": segment_popular,
        "global_popular": global_popular,
    }
    return metrics, payload


def train_recommender(
    transactions: pd.DataFrame,
    customer_table: pd.DataFrame,
    segment_assignments: pd.DataFrame,
    *,
    source_sha256: str,
    split_version: str,
    seed: int,
    neighbors: int,
    top_k: int,
    artifact_directory: Path,
    evidence_directory: Path,
    mlflow_tags: Mapping[str, str],
) -> pd.DataFrame:
    """Fit/evaluate the cosine item model using training customers only."""
    train_ids = set(
        customer_table.loc[
            customer_table["partition"].astype("string").eq("train"), "customer_id"
        ].astype("string")
    )
    assignments = segment_assignments.loc[
        segment_assignments["partition"].astype("string").eq("train")
    ].copy()
    lines = _positive_product_lines(transactions, train_ids)
    history, holdout = _leave_last_order_out(lines)
    started = time.perf_counter()
    metrics, evaluation_payload = _evaluate(
        history,
        holdout,
        assignments,
        neighbors=neighbors,
        top_k=top_k,
    )
    evaluation_seconds = time.perf_counter() - started

    full_matrix, customers, items, _, _ = _interaction_matrix(lines)
    final_neighbor_indices, final_similarities = _neighbor_arrays(full_matrix, neighbors=neighbors)
    segment_popular = _popular_by_segment(lines, assignments, limit=max(50, top_k))
    global_popular = list(
        lines.groupby("stock_code", observed=True)["quantity"]
        .sum()
        .sort_values(ascending=False)
        .index.astype("string")[: max(50, top_k)]
    )
    run_id = log_mlflow_run(
        run_name="recommender_item_to_item",
        family="recommender",
        parameters={
            "neighbors": neighbors,
            "top_k": top_k,
            "weighting": "log1p_quantity",
            "similarity": "cosine",
        },
        metrics={key: float(value) for key, value in metrics.items()},
        tags=dict(mlflow_tags),
        training_seconds=evaluation_seconds,
    )
    artifact_path = artifact_directory / "item_to_item_recommender.joblib"
    save_joblib_artifact(
        {
            "customer_ids": customers,
            "items": items,
            "neighbor_indices": final_neighbor_indices,
            "similarities": final_similarities,
            "segment_popular": segment_popular,
            "global_popular": global_popular,
            "source_sha256": source_sha256,
            "split_version": split_version,
            "seed": seed,
            "neighbors": neighbors,
            "top_k": top_k,
            "metrics": metrics,
            "mlflow_run_id": run_id,
            "held_out_test_accessed": False,
        },
        artifact_path,
    )
    del evaluation_payload
    evidence = pd.DataFrame(
        [
            {
                "model": "item_to_item_cosine",
                **metrics,
                "training_customers": len(train_ids),
                "eligible_evaluation_customers": int(metrics["evaluated_customers"]),
                "catalog_items": len(items),
                "neighbors": neighbors,
                "top_k": top_k,
                "weighting": "log1p_quantity",
                "evaluation_seconds": evaluation_seconds,
                "mlflow_run_id": run_id,
                "artifact": artifact_path.name,
                "held_out_test_accessed": False,
            }
        ]
    )
    evidence_directory.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(evidence_directory / "recommender_evaluation.csv", index=False)
    return evidence
