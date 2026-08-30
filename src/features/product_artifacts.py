"""Training-only product taxonomy, popularity, and reference-price artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from src.data.validation import DataValidationError


@dataclass(frozen=True)
class ProductArtifacts:
    """Frozen product statistics and taxonomy fitted from training history only."""

    taxonomy_version: str
    fit_cutoff: pd.Timestamp
    selected_clusters: int
    taxonomy: pd.DataFrame
    reference_prices: pd.DataFrame
    popularity: pd.DataFrame
    taxonomy_candidates: pd.DataFrame
    top_terms: dict[int, tuple[str, ...]]
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    normalizer: Normalizer
    clusterer: MiniBatchKMeans


def canonical_description_lookup(training_history: pd.DataFrame) -> pd.DataFrame:
    """Select modal, then latest, then lexical normalized description per StockCode."""
    eligible = training_history.loc[
        training_history["is_product"] & training_history["description_normalized"].notna(),
        ["stock_code", "description_normalized", "invoice_date"],
    ]
    if eligible.empty:
        raise DataValidationError("No training product descriptions are available for taxonomy")
    candidates = (
        eligible.groupby(["stock_code", "description_normalized"], observed=True)
        .agg(description_count=("invoice_date", "size"), latest_seen=("invoice_date", "max"))
        .reset_index()
        .sort_values(
            ["stock_code", "description_count", "latest_seen", "description_normalized"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
    )
    return candidates.drop_duplicates("stock_code", keep="first").reset_index(drop=True)


def _taxonomy_embeddings(
    descriptions: pd.Series,
    config: dict[str, Any],
    seed: int,
) -> tuple[TfidfVectorizer, TruncatedSVD, Normalizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        max_features=int(config["max_tfidf_features"]),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(descriptions.astype(str))
    components = min(int(config["svd_components"]), matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 1:
        raise DataValidationError("Product taxonomy needs at least two descriptions and terms")
    svd = TruncatedSVD(n_components=components, random_state=seed)
    reduced = svd.fit_transform(matrix)
    normalizer = Normalizer(copy=False)
    embeddings = normalizer.fit_transform(reduced)
    return vectorizer, svd, normalizer, embeddings


def _fit_taxonomy_candidates(
    embeddings: np.ndarray,
    *,
    candidate_clusters: list[int],
    sample_size: int,
    minimum_cluster_share: float,
    seed: int,
) -> tuple[MiniBatchKMeans, pd.DataFrame]:
    valid_candidates = sorted(
        {value for value in candidate_clusters if 2 <= value < len(embeddings)}
    )
    if not valid_candidates:
        raise DataValidationError(
            "No configured taxonomy cluster candidate fits the product population"
        )
    records: list[dict[str, float | int]] = []
    models: dict[int, MiniBatchKMeans] = {}
    for clusters in valid_candidates:
        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=seed,
            n_init=10,
            batch_size=min(1024, max(64, len(embeddings))),
        )
        labels = model.fit_predict(embeddings)
        counts = np.bincount(labels, minlength=clusters)
        minimum_share = float(counts.min() / counts.sum())
        silhouette = float(
            silhouette_score(
                embeddings,
                labels,
                sample_size=min(sample_size, len(embeddings)),
                random_state=seed,
            )
        )
        balance_penalty = max(0.0, minimum_cluster_share - minimum_share)
        records.append(
            {
                "clusters": clusters,
                "silhouette": silhouette,
                "minimum_cluster_share": minimum_share,
                "selection_score": silhouette - balance_penalty,
            }
        )
        models[clusters] = model
    metrics = pd.DataFrame(records).sort_values(
        ["selection_score", "clusters"], ascending=[False, True], kind="mergesort"
    )
    selected = int(metrics.iloc[0]["clusters"])
    return models[selected], metrics.sort_values("clusters", ignore_index=True)


def _cluster_top_terms(
    clusterer: MiniBatchKMeans,
    svd: TruncatedSVD,
    vectorizer: TfidfVectorizer,
) -> dict[int, tuple[str, ...]]:
    approximate_tfidf_centers = svd.inverse_transform(clusterer.cluster_centers_)
    names = vectorizer.get_feature_names_out()
    return {
        cluster_id: tuple(names[np.argsort(center)[-8:][::-1]].tolist())
        for cluster_id, center in enumerate(approximate_tfidf_centers)
    }


def fit_product_artifacts(
    transactions: pd.DataFrame,
    *,
    training_customer_ids: set[str],
    cutoff: pd.Timestamp,
    feature_config: dict[str, Any],
    seed: int,
) -> ProductArtifacts:
    """Fit every population-learned product artifact on training history only."""
    training = transactions.loc[
        transactions["customer_id"].isin(training_customer_ids)
        & transactions["invoice_date"].lt(cutoff)
    ].copy()
    descriptions = canonical_description_lookup(training)
    taxonomy_config = feature_config["taxonomy"]
    vectorizer, svd, normalizer, embeddings = _taxonomy_embeddings(
        descriptions["description_normalized"], taxonomy_config, seed
    )
    clusterer, candidates = _fit_taxonomy_candidates(
        embeddings,
        candidate_clusters=[int(value) for value in taxonomy_config["candidate_clusters"]],
        sample_size=int(taxonomy_config["silhouette_sample_size"]),
        minimum_cluster_share=float(taxonomy_config["min_cluster_share"]),
        seed=seed,
    )
    taxonomy = descriptions.copy()
    taxonomy["category_id"] = clusterer.predict(embeddings).astype("int16")
    taxonomy["taxonomy_version"] = str(taxonomy_config["version"])

    eligible_prices = training.loc[
        training["is_valid_merchandise"] & training["price"].gt(0),
        ["stock_code", "price"],
    ]
    reference_prices = (
        eligible_prices.groupby("stock_code", observed=True)["price"]
        .agg(reference_price="median", reference_observations="size")
        .reset_index()
    )
    minimum_observations = int(feature_config["markdown_min_observations"])
    reference_prices = reference_prices.loc[
        reference_prices["reference_observations"].ge(minimum_observations)
    ].reset_index(drop=True)

    product_events = training.loc[
        training["is_positive_purchase"], ["stock_code", "invoice"]
    ].drop_duplicates()
    popularity = (
        product_events.groupby("stock_code", observed=True)
        .size()
        .rename("training_order_frequency")
        .reset_index()
    )
    total = max(int(popularity["training_order_frequency"].sum()), 1)
    popularity["product_popularity"] = popularity["training_order_frequency"] / total
    rare_threshold = float(popularity["product_popularity"].quantile(0.10))
    popularity["is_rare_product"] = popularity["product_popularity"].le(rare_threshold)

    return ProductArtifacts(
        taxonomy_version=str(taxonomy_config["version"]),
        fit_cutoff=pd.Timestamp(cutoff),
        selected_clusters=clusterer.n_clusters,
        taxonomy=taxonomy,
        reference_prices=reference_prices,
        popularity=popularity,
        taxonomy_candidates=candidates,
        top_terms=_cluster_top_terms(clusterer, svd, vectorizer),
        vectorizer=vectorizer,
        svd=svd,
        normalizer=normalizer,
        clusterer=clusterer,
    )
