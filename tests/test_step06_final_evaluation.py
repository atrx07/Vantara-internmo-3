"""Guardrail tests for the single governed held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.validation import DataValidationError
from src.models.final_evaluate import (
    FREEZE_SENTINEL,
    _validate_final_lock,
    classification_metrics_at_threshold,
)


def _write_freeze(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "owner_step06_approval_received": True,
                "choices_frozen": True,
                "held_out_test_accessed": False,
                "final_test_status": "NOT_RUN",
            }
        ),
        encoding="utf-8",
    )


def test_binary_metrics_use_the_frozen_threshold() -> None:
    """Classification metrics must reflect the supplied immutable threshold."""
    metrics = classification_metrics_at_threshold(
        [0, 0, 1, 1], [0.10, 0.60, 0.40, 0.90], threshold=0.50
    )
    assert metrics == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.75,
        "confusion_matrix": [[1, 1], [1, 1]],
        "threshold": 0.5,
        "rows": 4,
    }


def test_final_lock_requires_status_sentinel(tmp_path: Path) -> None:
    """The final evaluator must refuse access before STATUS records the freeze gate."""
    freeze_path = tmp_path / "model_freeze.json"
    output = tmp_path / "final"
    output.mkdir()
    _write_freeze(freeze_path)
    (tmp_path / "STATUS.md").write_text("STEP 06 in progress\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="STATUS.md must contain"):
        _validate_final_lock(tmp_path, freeze_path, output)


def test_final_lock_refuses_a_second_attempt(tmp_path: Path) -> None:
    """An execution lock is sufficient to make any held-out rerun fail closed."""
    freeze_path = tmp_path / "model_freeze.json"
    output = tmp_path / "final"
    output.mkdir()
    _write_freeze(freeze_path)
    (tmp_path / "STATUS.md").write_text(FREEZE_SENTINEL, encoding="utf-8")
    (output / "final_evaluation_execution_lock.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DataValidationError, match="already started or completed"):
        _validate_final_lock(tmp_path, freeze_path, output)


def test_final_lock_accepts_a_frozen_untouched_state(tmp_path: Path) -> None:
    """The guard opens only when approval, freeze state, and STATUS agree."""
    freeze_path = tmp_path / "model_freeze.json"
    output = tmp_path / "final"
    output.mkdir()
    _write_freeze(freeze_path)
    (tmp_path / "STATUS.md").write_text(FREEZE_SENTINEL, encoding="utf-8")

    freeze = _validate_final_lock(tmp_path, freeze_path, output)

    assert freeze["choices_frozen"] is True
    assert freeze["held_out_test_accessed"] is False


def test_persisted_final_evidence_records_one_frozen_evaluation() -> None:
    """Tracked final evidence must remain complete and tied to the pre-test freeze."""
    root = Path(__file__).resolve().parents[1]
    evidence = root / "reports" / "final_evaluation"
    lock = json.loads(
        (evidence / "final_evaluation_execution_lock.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((evidence / "final_metrics.json").read_text(encoding="utf-8"))

    assert lock["evaluation_attempt"] == 1
    assert lock["choices_frozen"] is True
    assert metrics["freeze_sha256"] == lock["freeze_sha256"]
    assert metrics["choices_frozen_before_test"] is True
    assert metrics["held_out_test_accessed"] is True
    assert metrics["held_out_test_evaluations"] == 1
    assert {
        "churn",
        "clv",
        "next_purchase_lstm",
        "next_category",
        "autoencoder",
    }.issubset(metrics)
