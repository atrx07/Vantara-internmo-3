"""Integration checks for committed STEP 05 evidence and PyTorch artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.ann import ChurnANN, load_ann
from src.models.autoencoder import BehavioralAutoencoder, load_autoencoder
from src.models.purchase_lstm import PurchaseLSTM, load_purchase_lstm


def test_all_step05_artifacts_reload_safely_without_test_access() -> None:
    """All deep-learning artifacts use safe weights-only reload and carry the test lock."""
    directory = Path("models_artifacts/step05")
    assert {path.name for path in directory.glob("*.pt")} == {
        "churn_ann.pt",
        "purchase_lstm.pt",
        "behavioral_autoencoder.pt",
    }
    ann, ann_metadata = load_ann(directory / "churn_ann.pt")
    lstm, lstm_metadata = load_purchase_lstm(directory / "purchase_lstm.pt")
    autoencoder, autoencoder_metadata = load_autoencoder(directory / "behavioral_autoencoder.pt")
    assert isinstance(ann, ChurnANN)
    assert isinstance(lstm, PurchaseLSTM)
    assert isinstance(autoencoder, BehavioralAutoencoder)
    assert ann_metadata["held_out_test_accessed"] is False
    assert lstm_metadata["held_out_test_accessed"] is False
    assert autoencoder_metadata["held_out_test_accessed"] is False
    assert ann_metadata["feature_schema_version"] == "vantara-churn-features-v1"
    assert len(ann_metadata["feature_names"]) == 47


def test_step05_evidence_has_metrics_grouped_folds_and_reconstruction_details() -> None:
    """Tracked evidence contains every governed validation and reconstruction field."""
    evidence = Path("reports/deep_learning")
    summary = json.loads((evidence / "step05_summary.json").read_text(encoding="utf-8"))
    assert summary["step"] == "STEP 05"
    assert summary["held_out_test_accessed"] is False
    assert summary["feature_count"] == 47
    assert summary["mlflow_run_count"] == 8
    for model in ("ann", "lstm"):
        metrics = summary[model]["metrics"]
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
            assert name in metrics

    folds = pd.read_csv(evidence / "lstm_grouped_cv.csv")
    assert list(folds["fold"]) == [1, 2, 3, 4, 5]
    assert folds[["accuracy", "precision", "recall", "f1", "roc_auc"]].notna().all().all()
    assert folds["train_customers"].gt(0).all()
    assert folds["validation_customers"].gt(0).all()

    autoencoder = summary["autoencoder"]["metrics"]
    assert autoencoder["threshold_percentile"] == 99.0
    assert autoencoder["flagged_count"] == 8
    assert 0.0 < autoencoder["flagged_rate"] < 0.02
    contributions = pd.read_csv(evidence / "autoencoder_feature_contributions.csv")
    assert len(contributions) == 10
    assert contributions["mean_flagged_squared_error"].is_monotonic_decreasing

    reload_checks = json.loads(
        (evidence / "artifact_reload_smoke.json").read_text(encoding="utf-8")
    )
    assert len(reload_checks) == 3
    assert all(check["reload"] == "PASS" for check in reload_checks)


def test_step05_evidence_is_portable_and_mlflow_runs_finished() -> None:
    """Evidence contains no host path and all tracked run summaries are finished."""
    evidence = Path("reports/deep_learning")
    for path in evidence.glob("*"):
        if path.suffix not in {".csv", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "C:/Users/" not in text
        assert "C:\\Users\\" not in text
    runs = pd.read_csv(evidence / "mlflow_run_summary.csv")
    assert len(runs) == 8
    assert runs["status"].eq("FINISHED").all()
    assert set(runs["model_family"]) == {
        "deep_churn",
        "purchase_lstm_cv",
        "purchase_lstm",
        "autoencoder",
    }
