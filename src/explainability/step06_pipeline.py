"""Generate validation-only STEP 06 explainability assets for frozen models."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from src.data.validation import DataValidationError
from src.explainability.churn import plain_language_churn_explanation
from src.models.autoencoder import load_autoencoder
from src.models.purchase_lstm import build_rolling_sequences, load_purchase_lstm
from src.utils.config import load_config, resolve_project_path
from src.utils.logging import configure_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _class_one_explanation(explanation: shap.Explanation) -> shap.Explanation:
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        return shap.Explanation(
            values=values[:, :, 1],
            base_values=base_values[:, 1],
            data=explanation.data,
            feature_names=explanation.feature_names,
        )
    return explanation


def _save_churn_shap(
    pipeline: Pipeline,
    x_validation: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
    output: Path,
    local_max_features: int,
) -> tuple[pd.DataFrame, dict[str, int], shap.Explanation]:
    transformed = pipeline.named_steps["imputer"].transform(x_validation)
    raw_explanation = shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
    class_one = _class_one_explanation(raw_explanation)
    explanation = shap.Explanation(
        values=class_one.values,
        base_values=class_one.base_values,
        data=class_one.data,
        feature_names=list(x_validation.columns),
    )
    importance = pd.DataFrame(
        {
            "feature": x_validation.columns,
            "mean_absolute_shap": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    importance.to_csv(output / "churn_global_shap_importance.csv", index=False)
    shap.summary_plot(
        explanation.values,
        transformed,
        feature_names=list(x_validation.columns),
        show=False,
        plot_type="bar",
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output / "churn_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    representatives = {
        "low": int(np.argmin(probabilities)),
        "borderline": int(np.argmin(np.abs(probabilities - threshold))),
        "high": int(np.argmax(probabilities)),
    }
    local_rows: list[dict[str, Any]] = []
    for risk_label, index in representatives.items():
        local = explanation[index]
        shap.plots.waterfall(local, max_display=local_max_features, show=False)
        plt.tight_layout()
        plt.savefig(output / f"churn_shap_local_{risk_label}.png", dpi=160, bbox_inches="tight")
        plt.close()
        force = shap.force_plot(
            local.base_values,
            local.values,
            local.data,
            feature_names=list(x_validation.columns),
            matplotlib=False,
        )
        shap.save_html(str(output / f"churn_shap_force_{risk_label}.html"), force)
        for feature, value, contribution in zip(
            x_validation.columns, local.data, local.values, strict=True
        ):
            local_rows.append(
                {
                    "risk_profile": risk_label,
                    "row_index": index,
                    "customer_id": str(x_validation.index[index]),
                    "probability": float(probabilities[index]),
                    "feature": str(feature),
                    "feature_value": float(value),
                    "shap_value": float(contribution),
                }
            )
    pd.DataFrame(local_rows).to_csv(output / "churn_local_shap_values.csv", index=False)
    return importance, representatives, explanation


def _save_lime_and_plain_language(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    probabilities: np.ndarray,
    explanation: shap.Explanation,
    representatives: dict[str, int],
    *,
    threshold: float,
    output: Path,
    lime_features: int,
    seed: int,
) -> None:
    feature_names = list(x_train.columns)
    borderline_index = representatives["borderline"]
    lime_explainer = LimeTabularExplainer(
        x_train.to_numpy(dtype="float64"),
        feature_names=feature_names,
        class_names=["active", "churn"],
        mode="classification",
        random_state=seed,
        discretize_continuous=True,
    )

    def predict(values: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(pd.DataFrame(values, columns=feature_names))

    lime_explanation = lime_explainer.explain_instance(
        x_validation.iloc[borderline_index].to_numpy(dtype="float64"),
        predict,
        num_features=lime_features,
        labels=(1,),
    )
    lime_explanation.save_to_file(str(output / "churn_lime_borderline.html"))
    lime_rows = [
        {"condition": condition, "lime_weight": float(weight)}
        for condition, weight in lime_explanation.as_list(label=1)
    ]
    pd.DataFrame(lime_rows).to_csv(output / "churn_lime_borderline.csv", index=False)

    local_values = np.asarray(explanation.values[borderline_index])
    shap_order = np.argsort(np.abs(local_values))[::-1]
    shap_top = [feature_names[index] for index in shap_order[:lime_features]]
    lime_conditions = [row["condition"] for row in lime_rows]
    lime_features_resolved = [
        next(
            (name for name in sorted(feature_names, key=len, reverse=True) if name in condition),
            condition,
        )
        for condition in lime_conditions
    ]
    overlap = sorted(set(shap_top).intersection(lime_features_resolved))
    note = (
        "# SHAP–LIME borderline-customer comparison\n\n"
        f"The shared customer has validation risk `{probabilities[borderline_index]:.10f}` "
        f"against the frozen threshold `{threshold:.10f}`. SHAP and LIME share "
        f"`{len(overlap)}` of their top `{lime_features}` resolved features: "
        f"{', '.join(overlap) if overlap else 'none'}. Differences are expected because SHAP "
        "attributes the fitted forest prediction while LIME fits a local surrogate around one "
        "customer. Both outputs are descriptive and do not imply causality.\n"
    )
    (output / "shap_lime_comparison.md").write_text(note, encoding="utf-8")

    explanations: list[dict[str, Any]] = []
    for risk_label, index in representatives.items():
        contributions = np.asarray(explanation.values[index])
        positive = np.argsort(contributions)[::-1]
        negative = np.argsort(contributions)
        positive_names = [feature_names[i] for i in positive if contributions[i] > 0]
        negative_names = [feature_names[i] for i in negative if contributions[i] < 0]
        explanations.append(
            {
                "risk_profile": risk_label,
                "customer_id": str(x_validation.index[index]),
                "probability": float(probabilities[index]),
                "threshold": threshold,
                "text": plain_language_churn_explanation(
                    probability=float(probabilities[index]),
                    threshold=threshold,
                    positive_drivers=positive_names,
                    negative_drivers=negative_names,
                ),
            }
        )
    _write_json(output / "churn_plain_language_explanations.json", explanations)


def _save_pdp(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    top_features: Sequence[str],
    *,
    output: Path,
) -> None:
    PartialDependenceDisplay.from_estimator(
        pipeline,
        x_train.astype("float64"),
        features=list(top_features),
        kind="average",
        grid_resolution=30,
    )
    plt.tight_layout()
    plt.savefig(output / "churn_partial_dependence.png", dpi=160, bbox_inches="tight")
    plt.close()


def _save_clv_shap(root: Path, x_validation: pd.DataFrame, output: Path, sample_size: int) -> None:
    bundle = joblib.load(root / "models_artifacts" / "clv" / "production_clv.joblib")
    feature_names = list(bundle["metadata"]["feature_names"])
    outer = bundle["pipeline"]
    inner = outer.estimator_.regressor_
    sample = x_validation.loc[:, feature_names].head(sample_size)
    transformed = inner[:-1].transform(sample)
    model = inner.named_steps["model"]
    explanation = shap.LinearExplainer(model, transformed)(transformed)
    pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap_log1p_clv": np.abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_log1p_clv", ascending=False).to_csv(
        output / "clv_shap_importance.csv", index=False
    )


def _save_next_category_shap(
    root: Path, x_validation: pd.DataFrame, output: Path, sample_size: int
) -> None:
    bundle = joblib.load(
        root / "models_artifacts" / "next_category" / "next_category_lightgbm.joblib"
    )
    feature_names = list(bundle["metadata"]["feature_names"])
    pipeline = bundle["pipeline"]
    sample = x_validation.loc[:, feature_names].head(sample_size)
    transformed = pipeline.named_steps["imputer"].transform(sample)
    explanation = shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
    values = np.asarray(explanation.values)
    mean_absolute = np.abs(values).mean(axis=(0, 2))
    pd.DataFrame(
        {"feature": feature_names, "mean_absolute_multiclass_shap": mean_absolute}
    ).sort_values("mean_absolute_multiclass_shap", ascending=False).to_csv(
        output / "next_category_shap_importance.csv", index=False
    )


def _save_lstm_perturbation(
    root: Path,
    config: dict[str, Any],
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    split: pd.DataFrame,
    observation_end: pd.Timestamp,
    output: Path,
) -> None:
    sequences = build_rolling_sequences(
        transactions,
        taxonomy,
        split,
        observation_end=observation_end,
        config=config["deep_learning"]["lstm"],
    )
    validation_indices = np.flatnonzero(sequences.partitions == "validation")
    model, metadata = load_purchase_lstm(
        root / "models_artifacts" / "next_purchase" / "purchase_lstm.pt"
    )
    means = np.asarray(metadata["continuous_means"], dtype="float32")
    scales = np.asarray(metadata["continuous_scales"], dtype="float32")
    continuous = sequences.continuous[validation_indices].copy()
    lengths = sequences.lengths[validation_indices]
    for index, length in enumerate(lengths):
        continuous[index, : int(length)] = (continuous[index, : int(length)] - means) / scales
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(
                torch.from_numpy(continuous),
                torch.from_numpy(sequences.categories[validation_indices]),
                torch.from_numpy(lengths),
            )
        ).numpy()
    selected_position = int(np.argmax(probabilities))
    sequence_index = int(validation_indices[selected_position])
    length = int(sequences.lengths[sequence_index])
    base_probability = float(probabilities[selected_position])
    base_continuous = continuous[[selected_position]].copy()
    base_categories = sequences.categories[[sequence_index]].copy()
    rows: list[dict[str, Any]] = []
    for event_index in range(length):
        masked_continuous = base_continuous.copy()
        masked_categories = base_categories.copy()
        masked_continuous[0, event_index] = 0.0
        masked_categories[0, event_index] = 0
        with torch.no_grad():
            masked_probability = float(
                torch.sigmoid(
                    model(
                        torch.from_numpy(masked_continuous),
                        torch.from_numpy(masked_categories),
                        torch.tensor([length], dtype=torch.int64),
                    )
                )[0]
            )
        rows.append(
            {
                "customer_id": str(sequences.customer_ids[sequence_index]),
                "cutoff": str(sequences.cutoffs[sequence_index]),
                "event_position_oldest_zero": event_index,
                "event_recency_rank": length - event_index,
                "base_probability": base_probability,
                "masked_probability": masked_probability,
                "probability_delta": base_probability - masked_probability,
            }
        )
    pd.DataFrame(rows).sort_values("probability_delta", ascending=False).to_csv(
        output / "lstm_event_perturbation.csv", index=False
    )


def _save_autoencoder_explanation(root: Path, x_validation: pd.DataFrame, output: Path) -> None:
    model, metadata = load_autoencoder(
        root / "models_artifacts" / "autoencoder" / "behavioral_autoencoder.pt"
    )
    feature_names = list(metadata["feature_names"])
    transform = metadata["transform"]
    values = x_validation.loc[:, feature_names].to_numpy(dtype="float64")
    medians = np.asarray(transform["medians"], dtype="float64")
    means = np.asarray(transform["means"], dtype="float64")
    scales = np.asarray(transform["scales"], dtype="float64")
    scaled = ((np.where(np.isnan(values), medians, values) - means) / scales).astype("float32")
    with torch.no_grad():
        tensor = torch.from_numpy(scaled)
        reconstructed = model(tensor).numpy()
    errors = (reconstructed - scaled) ** 2
    selected = int(np.argmax(errors.mean(axis=1)))
    pd.DataFrame(
        {
            "customer_id": str(x_validation.index[selected]),
            "feature": feature_names,
            "scaled_value": scaled[selected],
            "reconstructed_value": reconstructed[selected],
            "squared_error_contribution": errors[selected],
        }
    ).sort_values("squared_error_contribution", ascending=False).to_csv(
        output / "autoencoder_local_contributions.csv", index=False
    )


def _save_recommender_and_segment_reasons(root: Path, taxonomy: pd.DataFrame, output: Path) -> None:
    recommender = joblib.load(
        root / "models_artifacts" / "recommendation" / "item_to_item_recommender.joblib"
    )
    descriptions = taxonomy.set_index("stock_code")["description_normalized"].astype(str).to_dict()
    items = list(recommender["items"])
    source_index = next(
        index
        for index, similarities in enumerate(recommender["similarities"])
        if np.any(np.asarray(similarities) > 0)
    )
    reasons = []
    for neighbor, similarity in zip(
        recommender["neighbor_indices"][source_index],
        recommender["similarities"][source_index],
        strict=True,
    ):
        if float(similarity) <= 0.0:
            continue
        source_code = str(items[source_index])
        recommended_code = str(items[int(neighbor)])
        reasons.append(
            {
                "source_stock_code": source_code,
                "source_description": descriptions.get(source_code, "unknown product"),
                "recommended_stock_code": recommended_code,
                "recommended_description": descriptions.get(recommended_code, "unknown product"),
                "cosine_similarity": float(similarity),
                "reason": f"Recommended because it is behaviorally similar to {source_code}.",
            }
        )
        if len(reasons) == 5:
            break
    _write_json(output / "recommender_reason_examples.json", reasons)
    profiles = pd.read_csv(root / "reports" / "modeling" / "segment_profiles.csv")
    profiles.to_csv(output / "segment_profile_explanations.csv", index=False)


def run_explainability(
    config_path: str | Path = "config/config.yaml",
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate all required explanation assets from validation/training evidence only."""
    config = load_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    freeze_config = config["model_freeze"]
    output = resolve_project_path(freeze_config["explainability_directory"], project_root=root)
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = (
        resolve_project_path(freeze_config["evidence_directory"], project_root=root)
        / "model_freeze.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("choices_frozen") or freeze.get("held_out_test_accessed"):
        raise DataValidationError("Explainability requires a validation-only frozen model record")
    churn_bundle = joblib.load(root / freeze["production_churn"]["artifact"])
    pipeline = churn_bundle["pipeline"]
    feature_names = list(churn_bundle["metadata"]["feature_names"])
    churn_table = pd.read_parquet(
        resolve_project_path(config["outputs"]["churn_features"], project_root=root)
    )
    indexed = churn_table.set_index("customer_id", drop=False)
    train = indexed.loc[indexed["partition"].astype("string").eq("train")]
    validation = indexed.loc[indexed["partition"].astype("string").eq("validation")]
    x_train = train.loc[:, feature_names]
    x_validation = validation.loc[:, feature_names]
    probabilities = pipeline.predict_proba(x_validation)[:, 1]
    xai_config = freeze_config["xai"]
    importance, representatives, explanation = _save_churn_shap(
        pipeline,
        x_validation,
        probabilities,
        threshold=float(freeze["production_churn"]["threshold"]),
        output=output,
        local_max_features=int(xai_config["local_max_features"]),
    )
    _save_lime_and_plain_language(
        pipeline,
        x_train,
        x_validation,
        probabilities,
        explanation,
        representatives,
        threshold=float(freeze["production_churn"]["threshold"]),
        output=output,
        lime_features=int(xai_config["lime_features"]),
        seed=int(config["project"]["random_seed"]),
    )
    _save_pdp(
        pipeline,
        x_train,
        list(importance.head(int(xai_config["pdp_features"]))["feature"]),
        output=output,
    )
    sample_size = int(xai_config["other_model_sample_size"])
    _save_clv_shap(root, validation, output, sample_size)
    _save_next_category_shap(root, validation, output, sample_size)
    transactions = pd.read_parquet(
        resolve_project_path(config["cleaning"]["interim_transactions"], project_root=root)
    )
    taxonomy = pd.read_parquet(
        resolve_project_path(config["outputs"]["product_taxonomy"], project_root=root)
    )
    split = pd.read_parquet(
        resolve_project_path(config["outputs"]["customer_split"], project_root=root)
    )
    metadata = json.loads(
        resolve_project_path(config["outputs"]["step02_metadata"], project_root=root).read_text(
            encoding="utf-8"
        )
    )
    _save_lstm_perturbation(
        root,
        config,
        transactions,
        taxonomy,
        split,
        pd.Timestamp(metadata["observation_end"]),
        output,
    )
    _save_autoencoder_explanation(root, validation, output)
    _save_recommender_and_segment_reasons(root, taxonomy, output)
    generated = sorted(path.name for path in output.iterdir() if path.is_file())
    summary = {
        "step": "STEP 06",
        "production_churn_model": freeze["production_churn"]["model"],
        "held_out_test_accessed": False,
        "representatives": {
            label: {
                "customer_id": str(x_validation.index[index]),
                "probability": float(probabilities[index]),
            }
            for label, index in representatives.items()
        },
        "top_global_features": list(importance.head(10)["feature"]),
        "generated_files": generated,
    }
    _write_json(output / "explainability_summary.json", summary)
    LOGGER.info(
        "STEP 06 validation-only explainability completed",
        extra={"event": "step06_explainability_completed", "files": len(generated) + 1},
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the explainability command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation-only explainability generation."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    configure_logging(str(config["logging"]["level"]))
    run_explainability(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
