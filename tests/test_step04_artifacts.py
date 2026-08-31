"""Integration checks for committed STEP 04 evidence and serialized artifacts."""

import json
from pathlib import Path

import joblib
import pandas as pd


def test_all_required_step04_model_artifacts_reload_with_test_lock() -> None:
    artifact_directory = Path("models_artifacts/step04")
    required = {
        "churn_logistic_regression.joblib",
        "churn_decision_tree.joblib",
        "churn_random_forest.joblib",
        "churn_xgboost.joblib",
        "churn_lightgbm.joblib",
        "churn_svm.joblib",
        "clv_ridge.joblib",
        "clv_xgboost_regressor.joblib",
        "segmentation_bundle.joblib",
        "next_category_lightgbm.joblib",
        "item_to_item_recommender.joblib",
    }
    assert {path.name for path in artifact_directory.glob("*.joblib")} == required
    for name in sorted(required):
        bundle = joblib.load(artifact_directory / name)
        metadata = bundle.get("metadata", bundle)
        assert metadata["held_out_test_accessed"] is False


def test_step04_evidence_has_all_models_metrics_and_mlflow_runs() -> None:
    evidence = Path("reports/modeling")
    churn = pd.read_csv(evidence / "churn_model_comparison.csv")
    assert set(churn["model"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
    }
    for metric in (
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "validation_confusion_matrix",
    ):
        assert churn[metric].notna().all()
    assert churn["held_out_test_accessed"].eq(False).all()

    clv = pd.read_csv(evidence / "clv_model_comparison.csv")
    assert set(clv["model"]) == {"ridge", "xgboost_regressor"}
    assert clv[["validation_mae", "validation_rmse", "validation_r2"]].notna().all().all()
    assert clv["held_out_test_accessed"].eq(False).all()

    runs = pd.read_csv(evidence / "mlflow_run_summary.csv")
    assert len(runs) == 23
    assert runs["status"].eq("FINISHED").all()
    assert set(runs["model_family"]) == {
        "churn",
        "clv",
        "segmentation",
        "next_category",
        "recommender",
    }


def test_step04_summary_and_reload_smoke_are_complete() -> None:
    evidence = Path("reports/modeling")
    summary = json.loads((evidence / "step04_summary.json").read_text(encoding="utf-8"))
    reload_checks = json.loads(
        (evidence / "artifact_reload_smoke.json").read_text(encoding="utf-8")
    )
    assert summary["step"] == "STEP 04"
    assert summary["held_out_test_accessed"] is False
    assert summary["feature_count"] == 47
    assert summary["mlflow_run_count"] == 23
    assert len(reload_checks) == 11
    assert all(check["reload"] == "PASS" for check in reload_checks)


def test_tracked_step04_evidence_has_no_machine_specific_absolute_path() -> None:
    for path in Path("reports/modeling").glob("*"):
        if path.suffix not in {".csv", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "C:/Users/" not in text
        assert "C:\\Users\\" not in text
