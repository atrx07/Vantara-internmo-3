"""Validation-only model-selection and threshold-freeze tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.validation import DataValidationError
from src.models.model_freeze import (
    consolidated_churn_comparison,
    ensure_freeze_is_open,
    select_churn_model,
    select_f2_threshold,
)


def test_locked_selection_prefers_recall_within_auc_tolerance() -> None:
    """Near-best eligible models are resolved by recall, exactly as governance requires."""
    comparison = pd.DataFrame(
        [
            {"model": "leader", "validation_roc_auc": 0.81, "validation_recall": 0.72},
            {"model": "recall_winner", "validation_roc_auc": 0.805, "validation_recall": 0.79},
            {"model": "outside_tolerance", "validation_roc_auc": 0.79, "validation_recall": 0.90},
            {"model": "ineligible", "validation_roc_auc": 0.90, "validation_recall": 0.69},
        ]
    )
    selected = select_churn_model(comparison, minimum_recall=0.70, roc_auc_tolerance=0.01)
    assert selected["model"] == "recall_winner"


def test_threshold_maximizes_f2_subject_to_recall_floor() -> None:
    """Threshold selection uses labels/probabilities supplied from validation only."""
    truth = [0, 0, 0, 1, 1, 1]
    probabilities = [0.05, 0.25, 0.45, 0.55, 0.70, 0.95]
    threshold, curve = select_f2_threshold(truth, probabilities, minimum_recall=0.70, beta=2.0)
    selected = curve.iloc[(curve["threshold"] - threshold).abs().argmin()]
    assert selected["recall"] >= 0.70
    assert selected["f_beta"] == curve.loc[curve["recall"].ge(0.70), "f_beta"].max()


def test_consolidated_comparison_is_generated_from_evidence(tmp_path: Path) -> None:
    """The seven-model table combines tracked classical and ANN evidence without test rows."""
    classical = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "validation_accuracy": 0.7,
                "validation_precision": 0.7,
                "validation_recall": 0.8,
                "validation_f1": 0.75,
                "validation_roc_auc": 0.81,
                "validation_confusion_matrix": "[[1, 2], [3, 4]]",
                "artifact": "churn_random_forest.joblib",
                "held_out_test_accessed": False,
            }
        ]
    )
    classical_path = tmp_path / "classical.csv"
    classical.to_csv(classical_path, index=False)
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "held_out_test_accessed": False,
                "ann": {
                    "artifact": "churn_ann.pt",
                    "metrics": {
                        "accuracy": 0.6,
                        "precision": 0.6,
                        "recall": 0.7,
                        "f1": 0.65,
                        "roc_auc": 0.75,
                        "confusion_matrix": [[1, 1], [1, 1]],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    comparison = consolidated_churn_comparison(classical_path, ann_path)
    assert list(comparison["model"]) == ["random_forest", "ann"]
    assert not comparison["held_out_test_accessed"].any()


def test_model_freeze_refuses_post_final_replacement(tmp_path: Path) -> None:
    """Once an evaluation lock exists, production choices cannot be regenerated."""
    (tmp_path / "final_evaluation_execution_lock.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DataValidationError, match="immutable after final"):
        ensure_freeze_is_open(tmp_path)
