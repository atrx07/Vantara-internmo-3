"""STEP 04 supervised-model contract and leakage-boundary tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.clv import stratified_clv_folds
from src.models.common import (
    classification_metrics,
    regression_metrics,
    supervised_partitions,
)


def test_supervised_partitions_never_expose_final_test_values() -> None:
    table = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E", "F"],
            "partition": ["train", "train", "validation", "validation", "test", "test"],
            "feature": [1.0, 2.0, 3.0, 4.0, 999.0, -999.0],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )
    original = supervised_partitions(table, feature_names=["feature"], target_name="target")
    attacked = table.copy()
    attacked.loc[attacked["partition"].eq("test"), ["feature", "target"]] = [123456.0, 1]
    repeated = supervised_partitions(attacked, feature_names=["feature"], target_name="target")

    pd.testing.assert_frame_equal(original.x_train, repeated.x_train)
    pd.testing.assert_frame_equal(original.x_validation, repeated.x_validation)
    pd.testing.assert_series_equal(original.y_train, repeated.y_train)
    pd.testing.assert_series_equal(original.y_validation, repeated.y_validation)
    assert set(original.train_customer_ids) == {"A", "B"}
    assert set(original.validation_customer_ids) == {"C", "D"}


def test_governed_metric_helpers_report_required_values() -> None:
    classification = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert set(classification) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "threshold",
    }
    assert classification["confusion_matrix"] == [[1, 1], [0, 2]]
    assert classification["threshold"] == 0.5

    regression = regression_metrics([0.0, 10.0, 20.0], [0.0, 8.0, 22.0])
    assert set(regression) == {"mae", "rmse", "r2"}
    assert regression["mae"] > 0
    assert regression["rmse"] > 0


def test_clv_cv_is_five_fold_training_only_and_disjoint() -> None:
    target = pd.Series(np.linspace(0.0, 1000.0, 100))
    folds = stratified_clv_folds(target, seed=42)
    assert len(folds) == 5
    validation_seen: set[int] = set()
    for training, validation in folds:
        assert set(training).isdisjoint(validation)
        validation_seen.update(int(value) for value in validation)
    assert validation_seen == set(range(len(target)))


def test_step04_artifact_metadata_declares_no_final_test_access() -> None:
    summary_path = Path("reports/modeling/step04_summary.json")
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["held_out_test_accessed"] is False
