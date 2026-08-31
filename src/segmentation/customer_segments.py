"""Training-only K-Means/GMM selection, profiling, and PCA evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.validation import DataValidationError
from src.models.common import log_mlflow_run, save_joblib_artifact, timed_fit


def _business_labels(profile: pd.DataFrame) -> dict[int, str]:
    score = (
        -profile["recency_days"].rank(pct=True)
        + profile["frequency_orders"].rank(pct=True)
        + profile["net_spend"].rank(pct=True)
    )
    ordered = list(score.sort_values(ascending=False).index.astype("int64"))
    names = [
        "Champions",
        "Loyal High Value",
        "Established",
        "Developing",
        "Occasional",
        "At Risk",
        "Dormant",
        "Lowest Engagement",
    ]
    return {segment: names[index] for index, segment in enumerate(ordered)}


def _profiles(
    frame: pd.DataFrame,
    *,
    segment_column: str,
    features: Sequence[str],
    algorithm: str,
) -> pd.DataFrame:
    grouped = frame.groupby(segment_column, observed=True)
    profile = grouped[list(features)].mean().reset_index()
    counts = grouped.size().rename("customer_count").reset_index()
    profile = profile.merge(counts, on=segment_column, validate="one_to_one")
    labels = _business_labels(profile.set_index(segment_column))
    profile["business_label"] = profile[segment_column].map(labels)
    profile.insert(0, "algorithm", algorithm)
    return profile


def train_segmentation_models(
    table: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    kmeans_candidates: Sequence[int],
    gmm_candidates: Sequence[int],
    split_version: str,
    source_sha256: str,
    seed: int,
    artifact_directory: Path,
    evidence_directory: Path,
    mlflow_tags: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select K-Means and GMM on train rows and assign validation without test use."""
    required = {"customer_id", "partition", *feature_names}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise DataValidationError(f"Segmentation table is missing columns: {missing}")
    partition = table["partition"].astype("string")
    train = table.loc[partition.eq("train")].copy()
    validation = table.loc[partition.eq("validation")].copy()
    if train.empty or validation.empty:
        raise DataValidationError("Segmentation requires non-empty train and validation rows")

    preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    train_scaled, preprocessing_seconds = timed_fit(
        preprocessor.fit_transform, train.loc[:, feature_names]
    )
    validation_scaled = preprocessor.transform(validation.loc[:, feature_names])
    metric_rows: list[dict[str, Any]] = []
    kmeans_models: dict[int, KMeans] = {}
    gmm_models: dict[int, GaussianMixture] = {}

    for clusters in kmeans_candidates:
        model, seconds = timed_fit(
            KMeans(n_clusters=int(clusters), n_init=20, random_state=seed).fit,
            train_scaled,
        )
        labels = model.labels_
        metrics = {
            "silhouette": float(silhouette_score(train_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(train_scaled, labels)),
            "inertia": float(model.inertia_),
        }
        run_id = log_mlflow_run(
            run_name=f"segmentation_kmeans_k{clusters}",
            family="segmentation",
            parameters={"algorithm": "kmeans", "clusters": int(clusters), "n_init": 20},
            metrics=metrics,
            tags=dict(mlflow_tags),
            training_seconds=seconds,
        )
        metric_rows.append(
            {
                "algorithm": "kmeans",
                "components": int(clusters),
                **metrics,
                "bic": np.nan,
                "training_seconds": seconds,
                "mlflow_run_id": run_id,
                "held_out_test_accessed": False,
            }
        )
        kmeans_models[int(clusters)] = model

    for components in gmm_candidates:
        model, seconds = timed_fit(
            GaussianMixture(
                n_components=int(components),
                covariance_type="diag",
                n_init=5,
                reg_covar=1e-5,
                random_state=seed,
            ).fit,
            train_scaled,
        )
        labels = model.predict(train_scaled)
        metrics = {
            "silhouette": float(silhouette_score(train_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(train_scaled, labels)),
            "bic": float(model.bic(train_scaled)),
        }
        run_id = log_mlflow_run(
            run_name=f"segmentation_gmm_k{components}",
            family="segmentation",
            parameters={
                "algorithm": "gmm",
                "components": int(components),
                "covariance_type": "diag",
            },
            metrics=metrics,
            tags=dict(mlflow_tags),
            training_seconds=seconds,
        )
        metric_rows.append(
            {
                "algorithm": "gmm",
                "components": int(components),
                "inertia": np.nan,
                **metrics,
                "training_seconds": seconds,
                "mlflow_run_id": run_id,
                "held_out_test_accessed": False,
            }
        )
        gmm_models[int(components)] = model

    metrics_frame = pd.DataFrame(metric_rows)
    kmeans_row = (
        metrics_frame.loc[metrics_frame["algorithm"].eq("kmeans")]
        .sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
        .iloc[0]
    )
    gmm_row = (
        metrics_frame.loc[metrics_frame["algorithm"].eq("gmm")]
        .sort_values(["bic", "silhouette"], ascending=[True, False])
        .iloc[0]
    )
    selected_kmeans = kmeans_models[int(kmeans_row["components"])]
    selected_gmm = gmm_models[int(gmm_row["components"])]

    combined = pd.concat([train, validation], ignore_index=True)
    combined_scaled = np.vstack([train_scaled, validation_scaled])
    combined["kmeans_segment"] = selected_kmeans.predict(combined_scaled).astype("int16")
    combined["gmm_segment"] = selected_gmm.predict(combined_scaled).astype("int16")
    kmeans_profiles = _profiles(
        train.assign(kmeans_segment=selected_kmeans.labels_),
        segment_column="kmeans_segment",
        features=feature_names,
        algorithm="kmeans",
    )
    gmm_profiles = _profiles(
        train.assign(gmm_segment=selected_gmm.predict(train_scaled)),
        segment_column="gmm_segment",
        features=feature_names,
        algorithm="gmm",
    )
    profiles = pd.concat([kmeans_profiles, gmm_profiles], ignore_index=True)

    pca = PCA(n_components=2, random_state=seed).fit(train_scaled)
    coordinates = pca.transform(combined_scaled)
    pca_frame = pd.DataFrame(
        {
            "partition": combined["partition"].astype("string"),
            "pca_1": coordinates[:, 0],
            "pca_2": coordinates[:, 1],
            "kmeans_segment": combined["kmeans_segment"],
            "gmm_segment": combined["gmm_segment"],
        }
    )
    pca_sample = pca_frame.sample(n=min(1000, len(pca_frame)), random_state=seed).sort_index()

    artifact_path = artifact_directory / "segmentation_bundle.joblib"
    save_joblib_artifact(
        {
            "preprocessor": preprocessor,
            "kmeans": selected_kmeans,
            "gmm": selected_gmm,
            "pca": pca,
            "feature_names": list(feature_names),
            "split_version": split_version,
            "source_sha256": source_sha256,
            "seed": seed,
            "kmeans_labels": dict(
                zip(
                    kmeans_profiles["kmeans_segment"].astype("int64"),
                    kmeans_profiles["business_label"],
                    strict=True,
                )
            ),
            "gmm_labels": dict(
                zip(
                    gmm_profiles["gmm_segment"].astype("int64"),
                    gmm_profiles["business_label"],
                    strict=True,
                )
            ),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "preprocessing_seconds": preprocessing_seconds,
            "held_out_test_accessed": False,
        },
        artifact_path,
    )
    evidence_directory.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(evidence_directory / "segmentation_model_selection.csv", index=False)
    profiles.to_csv(evidence_directory / "segment_profiles.csv", index=False)
    pca_sample.to_csv(evidence_directory / "segmentation_pca_sample.csv", index=False)
    assignments = combined[["customer_id", "partition", "kmeans_segment", "gmm_segment"]].copy()
    return metrics_frame, profiles, assignments
