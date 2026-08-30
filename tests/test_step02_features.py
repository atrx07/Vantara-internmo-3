"""Product artifact, customer feature, and preprocessing leakage tests."""

import numpy as np
import pandas as pd

from src.data.splits import create_customer_split
from src.features.customer_features import build_customer_features
from src.features.preprocessing import fit_preprocessing_contracts
from src.features.product_artifacts import (
    ProductArtifacts,
    canonical_description_lookup,
    fit_product_artifacts,
)


def _feature_config() -> dict[str, object]:
    return {
        "markdown_price_ratio": 0.90,
        "markdown_min_observations": 1,
        "taxonomy": {
            "version": "test-taxonomy-v1",
            "candidate_clusters": [2, 3],
            "max_tfidf_features": 100,
            "svd_components": 5,
            "silhouette_sample_size": 100,
            "min_cluster_share": 0.01,
        },
    }


def _artifacts(
    transactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    training_ids: set[str],
) -> ProductArtifacts:
    return fit_product_artifacts(
        transactions,
        training_customer_ids=training_ids,
        cutoff=cutoff,
        feature_config=_feature_config(),
        seed=42,
    )


def test_canonical_description_lookup_is_deterministic(step02_transactions: pd.DataFrame) -> None:
    first = canonical_description_lookup(step02_transactions)
    shuffled = step02_transactions.sample(frac=1.0, random_state=42)
    second = canonical_description_lookup(shuffled)

    pd.testing.assert_frame_equal(first, second)


def test_future_rows_do_not_change_fitted_taxonomy_reference_prices_or_frequency(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-05-01")
    training_ids = {f"CUST{index}" for index in range(6)}
    baseline = _artifacts(step02_transactions, cutoff=cutoff, training_ids=training_ids)
    future = step02_transactions.iloc[[0]].copy()
    future["invoice"] = "FUTURE"
    future["invoice_date"] = cutoff + pd.Timedelta(days=1)
    future["price"] = 9999.0
    future["signed_merchandise_value"] = future["quantity"].astype(float) * future["price"]
    future["gross_positive_value"] = future["signed_merchandise_value"]
    attacked = _artifacts(
        pd.concat([step02_transactions, future], ignore_index=True),
        cutoff=cutoff,
        training_ids=training_ids,
    )

    pd.testing.assert_frame_equal(baseline.taxonomy, attacked.taxonomy)
    pd.testing.assert_frame_equal(baseline.reference_prices, attacked.reference_prices)
    pd.testing.assert_frame_equal(baseline.popularity, attacked.popularity)
    pd.testing.assert_frame_equal(baseline.taxonomy_candidates, attacked.taxonomy_candidates)


def test_future_transaction_insertion_does_not_change_historical_features(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-05-01")
    customers = [f"CUST{index}" for index in range(8)]
    training_ids = {f"CUST{index}" for index in range(6)}
    artifacts = _artifacts(step02_transactions, cutoff=cutoff, training_ids=training_ids)
    baseline, _ = build_customer_features(
        step02_transactions,
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=training_ids,
        trend_window_days=90,
        markdown_price_ratio=0.90,
    )
    future = step02_transactions.iloc[[0]].copy()
    future["invoice"] = "AFTER-CUTOFF"
    future["invoice_date"] = cutoff + pd.Timedelta(seconds=1)
    future["quantity"] = 500
    future["signed_merchandise_value"] = 500 * future["price"]
    future["gross_positive_value"] = future["signed_merchandise_value"]
    attacked, _ = build_customer_features(
        pd.concat([step02_transactions, future], ignore_index=True),
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=training_ids,
        trend_window_days=90,
        markdown_price_ratio=0.90,
    )

    pd.testing.assert_frame_equal(baseline, attacked)


def test_customer_features_have_full_affinity_and_training_fitted_engagement(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-05-01")
    customers = [f"CUST{index}" for index in range(8)]
    training_ids = {f"CUST{index}" for index in range(6)}
    artifacts = _artifacts(step02_transactions, cutoff=cutoff, training_ids=training_ids)
    features, engagement = build_customer_features(
        step02_transactions,
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=training_ids,
        trend_window_days=90,
        markdown_price_ratio=0.90,
    )

    affinity = [name for name in features if name.startswith("category_affinity_")]
    assert len(affinity) == artifacts.selected_clusters + 1
    assert np.allclose(features[affinity].sum(axis=1), 1.0)
    assert features["engagement_score"].between(0.0, 100.0).all()
    assert len(engagement.recency) == len(training_ids)


def test_validation_population_cannot_change_engagement_fit(
    step02_transactions: pd.DataFrame,
) -> None:
    cutoff = pd.Timestamp("2021-05-01")
    customers = [f"CUST{index}" for index in range(8)]
    training_ids = {f"CUST{index}" for index in range(6)}
    artifacts = _artifacts(step02_transactions, cutoff=cutoff, training_ids=training_ids)
    _, baseline = build_customer_features(
        step02_transactions,
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=training_ids,
        trend_window_days=90,
        markdown_price_ratio=0.90,
    )
    attack = step02_transactions.copy()
    validation_index = attack.loc[attack["customer_id"] == "CUST7"].index[0]
    attack.loc[validation_index, "gross_positive_value"] = 999999.0
    attack.loc[validation_index, "signed_merchandise_value"] = 999999.0
    _, attacked = build_customer_features(
        attack,
        customer_ids=customers,
        cutoff=cutoff,
        artifacts=artifacts,
        training_customer_ids=training_ids,
        trend_window_days=90,
        markdown_price_ratio=0.90,
    )

    np.testing.assert_array_equal(baseline.recency, attacked.recency)
    np.testing.assert_array_equal(baseline.frequency, attacked.frequency)
    np.testing.assert_array_equal(baseline.monetary, attacked.monetary)


def test_validation_and_test_rows_cannot_affect_fitted_preprocessors() -> None:
    customers = [f"CUST{index}" for index in range(20)]
    split = create_customer_split(
        customers,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=42,
        version="test-v1",
    )
    features = pd.DataFrame(
        {
            "customer_id": pd.Series(customers, dtype="string"),
            "cutoff_timestamp": pd.Timestamp("2021-05-01"),
            "feature_a": np.arange(20, dtype=float),
            "feature_b": np.arange(20, dtype=float) ** 2,
        }
    )
    baseline = fit_preprocessing_contracts(features, split)
    attacked_features = features.copy()
    non_train = set(split.loc[split["partition"] != "train", "customer_id"].astype(str))
    attacked_features.loc[attacked_features["customer_id"].isin(non_train), "feature_a"] = 1e12
    attacked = fit_preprocessing_contracts(attacked_features, split)

    baseline_scaler = baseline.scaled.pipeline.named_steps["scaler"]
    attacked_scaler = attacked.scaled.pipeline.named_steps["scaler"]
    np.testing.assert_array_equal(baseline_scaler.mean_, attacked_scaler.mean_)
    np.testing.assert_array_equal(baseline_scaler.scale_, attacked_scaler.scale_)
