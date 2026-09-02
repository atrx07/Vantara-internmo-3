"""One-time loading, compatibility validation, and inference for serving artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from src.explainability.churn import plain_language_churn_explanation
from src.models.autoencoder import load_autoencoder
from src.models.purchase_lstm import load_purchase_lstm


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a serving artifact is missing or violates the frozen contract."""


@dataclass(frozen=True)
class ScoreResult:
    """Complete model output generated from one governed customer feature row."""

    churn_probability: float
    churn_label: bool
    churn_threshold: float
    predicted_clv_180d: float
    next_purchase_probability: float | None
    next_category_id: str
    next_category_probability: float
    anomaly_score: float
    anomaly_flag: bool
    segment_id: int
    segment_name: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactRegistry:
    """Load frozen production artifacts once and expose deterministic scoring methods."""

    def __init__(self, project_root: Path, artifact_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.artifact_root = artifact_root.resolve()
        freeze_path = self.project_root / "reports" / "model_freeze" / "model_freeze.json"
        if not freeze_path.is_file():
            raise ArtifactCompatibilityError(f"Missing model freeze record: {freeze_path}")
        self.freeze: dict[str, Any] = json.loads(freeze_path.read_text(encoding="utf-8"))
        self._validate_inventory()
        self.churn = joblib.load(self.artifact_root / "churn" / "production_churn.joblib")
        self.clv = joblib.load(self.artifact_root / "clv" / "production_clv.joblib")
        self.next_category = joblib.load(
            self.artifact_root / "next_category" / "next_category_lightgbm.joblib"
        )
        self.segmentation = joblib.load(
            self.artifact_root / "segmentation" / "segmentation_bundle.joblib"
        )
        self.recommender = joblib.load(
            self.artifact_root / "recommendation" / "item_to_item_recommender.joblib"
        )
        references = self.artifact_root / "serving_reference"
        self.product_taxonomy = pd.read_parquet(
            self.artifact_root / "product_taxonomy" / "product_taxonomy.parquet"
        )
        self.reference_prices = pd.read_parquet(references / "product_reference_prices.parquet")
        self.product_popularity = pd.read_parquet(references / "product_frequency_encoding.parquet")
        self.preprocessing_contracts = joblib.load(references / "preprocessing_contracts.joblib")
        self.purchase_lstm, self.purchase_lstm_metadata = load_purchase_lstm(
            self.artifact_root / "next_purchase" / "purchase_lstm.pt"
        )
        self.autoencoder, self.autoencoder_metadata = load_autoencoder(
            self.artifact_root / "autoencoder" / "behavioral_autoencoder.pt"
        )
        self.feature_names = tuple(str(value) for value in self.churn["metadata"]["feature_names"])
        self._validate_metadata()
        churn_pipeline = self.churn["pipeline"]
        self.churn_explainer = shap.TreeExplainer(churn_pipeline.named_steps["model"])

    def _validate_inventory(self) -> None:
        inventory = self.freeze.get("serving_artifacts", [])
        if not inventory:
            raise ArtifactCompatibilityError("Model freeze contains no serving inventory")
        for item in inventory:
            relative = Path(str(item["path"]))
            parts = relative.parts
            if parts and parts[0] == "models_artifacts":
                relative = Path(*parts[1:])
            path = self.artifact_root / relative
            if not path.is_file():
                raise ArtifactCompatibilityError(f"Missing serving artifact: {path}")
            if _sha256(path) != str(item["sha256"]):
                raise ArtifactCompatibilityError(f"Serving artifact hash mismatch: {path}")

    def _validate_metadata(self) -> None:
        expected_schema = str(self.freeze["feature_schema_version"])
        expected_source = str(self.freeze["source_sha256"])
        bundles = (self.churn, self.clv, self.next_category)
        for bundle in bundles:
            metadata = bundle["metadata"]
            if str(metadata["feature_schema_version"]) != expected_schema:
                raise ArtifactCompatibilityError("Serving artifact feature schema mismatch")
            if str(metadata["source_sha256"]) != expected_source:
                raise ArtifactCompatibilityError("Serving artifact source hash mismatch")
            if tuple(str(value) for value in metadata["feature_names"]) != self.feature_names:
                raise ArtifactCompatibilityError("Serving artifact feature order mismatch")
        if str(self.segmentation["source_sha256"]) != expected_source:
            raise ArtifactCompatibilityError("Segmentation artifact source hash mismatch")

    def score(
        self,
        feature_payload: dict[str, float],
        sequence_payload: dict[str, object] | None,
    ) -> ScoreResult:
        """Score every available frozen model from persisted server-owned inputs."""
        missing = sorted(set(self.feature_names).difference(feature_payload))
        if missing:
            raise ArtifactCompatibilityError(f"Customer feature payload is missing: {missing}")
        features = pd.DataFrame(
            [[feature_payload[name] for name in self.feature_names]], columns=self.feature_names
        )
        churn_probability = float(self.churn["pipeline"].predict_proba(features)[0, 1])
        threshold = float(self.freeze["production_churn"]["threshold"])
        clv = max(float(self.clv["pipeline"].predict(features)[0]), 0.0)

        category_probabilities = self.next_category["pipeline"].predict_proba(features)[0]
        category_index = int(np.argmax(category_probabilities))
        category_id = str(
            self.next_category["label_encoder"].inverse_transform([category_index])[0]
        )

        auto_features = list(self.autoencoder_metadata["feature_names"])
        values = features.loc[:, auto_features].to_numpy(dtype="float64")
        transform = self.autoencoder_metadata["transform"]
        medians = np.asarray(transform["medians"], dtype="float64")
        means = np.asarray(transform["means"], dtype="float64")
        scales = np.asarray(transform["scales"], dtype="float64")
        scaled = ((np.where(np.isnan(values), medians, values) - means) / scales).astype("float32")
        with torch.no_grad():
            reconstructed = self.autoencoder(torch.from_numpy(scaled)).numpy()
        anomaly_score = float(np.square(reconstructed - scaled).mean())
        anomaly_threshold = float(self.freeze["production_autoencoder"]["threshold"])

        segment_features = list(self.segmentation["feature_names"])
        segment_scaled = self.segmentation["preprocessor"].transform(
            features.loc[:, segment_features]
        )
        segment_id = int(self.segmentation["kmeans"].predict(segment_scaled)[0])
        segment_name = str(
            self.segmentation["kmeans_labels"].get(segment_id, f"Segment {segment_id}")
        )

        return ScoreResult(
            churn_probability=churn_probability,
            churn_label=churn_probability >= threshold,
            churn_threshold=threshold,
            predicted_clv_180d=clv,
            next_purchase_probability=self._score_sequence(sequence_payload),
            next_category_id=category_id,
            next_category_probability=float(category_probabilities[category_index]),
            anomaly_score=anomaly_score,
            anomaly_flag=anomaly_score >= anomaly_threshold,
            segment_id=segment_id,
            segment_name=segment_name,
        )

    def _score_sequence(self, payload: dict[str, object] | None) -> float | None:
        if payload is None:
            return None
        continuous = np.asarray(payload["continuous"], dtype="float32")[None, :, :]
        categories = np.asarray(payload["categories"], dtype="int64")[None, :]
        length = np.asarray([int(payload["length"])], dtype="int64")
        means = np.asarray(self.purchase_lstm_metadata["continuous_means"], dtype="float32")
        scales = np.asarray(self.purchase_lstm_metadata["continuous_scales"], dtype="float32")
        continuous[0, : length[0]] = (continuous[0, : length[0]] - means) / scales
        with torch.no_grad():
            logit = self.purchase_lstm(
                torch.from_numpy(continuous),
                torch.from_numpy(categories),
                torch.from_numpy(length),
            )
        return float(torch.sigmoid(logit)[0])

    def explain(self, feature_payload: dict[str, float], probability: float) -> dict[str, Any]:
        """Generate one-customer TreeSHAP drivers and deterministic business language."""
        features = pd.DataFrame(
            [[feature_payload[name] for name in self.feature_names]], columns=self.feature_names
        )
        transformed = self.churn["pipeline"].named_steps["imputer"].transform(features)
        explanation = self.churn_explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        local = values[0]
        upward = [
            self.feature_names[index] for index in np.argsort(local)[::-1] if local[index] > 0
        ][:3]
        downward = [self.feature_names[index] for index in np.argsort(local) if local[index] < 0][
            :3
        ]
        threshold = float(self.freeze["production_churn"]["threshold"])
        return {
            "probability": probability,
            "threshold": threshold,
            "positive_drivers": upward,
            "negative_drivers": downward,
            "text": plain_language_churn_explanation(
                probability=probability,
                threshold=threshold,
                positive_drivers=upward,
                negative_drivers=downward,
            ),
            "method": "TreeSHAP",
            "causal": False,
        }

    def safe_metadata(self) -> dict[str, Any]:
        """Return model names, versions, thresholds, schemas, and safe final evidence."""
        metrics_path = self.project_root / "reports" / "final_evaluation" / "final_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "freeze_version": self.freeze["freeze_version"],
            "feature_schema_version": self.freeze["feature_schema_version"],
            "feature_count": self.freeze["feature_count"],
            "split_version": self.freeze["split_version"],
            "churn": {
                "model": self.freeze["production_churn"]["model"],
                "version": self.churn["metadata"].get(
                    "model_version", "vantara-churn-production-v1"
                ),
                "threshold": self.freeze["production_churn"]["threshold"],
                "held_out_metrics": metrics["churn"],
            },
            "clv": {
                "model": self.freeze["production_clv"]["model"],
                "business_label": "Predicted 180-Day Customer Value",
                "held_out_metrics": metrics["clv"],
            },
            "next_purchase": {
                "model": self.freeze["production_next_purchase"]["model"],
                "horizon_days": 30,
            },
            "next_category": {"model": self.freeze["production_next_category"]["model"]},
            "autoencoder": {
                "model": self.freeze["production_autoencoder"]["model"],
                "threshold": self.freeze["production_autoencoder"]["threshold"],
                "interpretation": "manual-review anomaly candidate; not confirmed fraud",
            },
        }
