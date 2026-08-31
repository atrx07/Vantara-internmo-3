"""Generate the governed STEP 04 model-experiments consumer notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_model_experiments_notebook() -> nbformat.NotebookNode:
    """Build a report-consumer notebook with no notebook-owned model training."""
    cells = [
        new_markdown_cell(
            "# Vantara — 03 Model Experiments\n\n"
            "This notebook is a thin analysis consumer of STEP 04 evidence produced by "
            "`src.models.step04_pipeline`. It does not fit models or access the final "
            "held-out test."
        ),
        new_code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "ROOT = Path.cwd()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n\n"
            "from src.models.common import load_feature_schema  # noqa: E402\n\n"
            "REPORTS = ROOT / 'reports/modeling'\n"
            "schema = load_feature_schema(ROOT / 'models_artifacts/churn_feature_schema.json')\n"
            "summary = json.loads((REPORTS / 'step04_summary.json').read_text())\n"
            "{\n"
            "    'schema_version': schema['schema_version'],\n"
            "    'feature_count': schema['feature_count'],\n"
            "    'held_out_test_accessed': summary['held_out_test_accessed'],\n"
            "}"
        ),
        new_markdown_cell("## Six classical churn models — training CV and validation evidence"),
        new_code_cell(
            "churn = pd.read_csv(REPORTS / 'churn_model_comparison.csv')\n"
            "display(churn[[\n"
            "    'model', 'cv_roc_auc_mean', 'validation_accuracy', 'validation_precision',\n"
            "    'validation_recall', 'validation_f1', 'validation_roc_auc',\n"
            "    'validation_confusion_matrix'\n"
            "]])"
        ),
        new_markdown_cell("## Predicted 180-Day Customer Value — Ridge and XGBRegressor"),
        new_code_cell(
            "clv = pd.read_csv(REPORTS / 'clv_model_comparison.csv')\n"
            "display(clv[[\n"
            "    'model', 'cv_mae_mean', 'cv_rmse_mean', 'cv_r2_mean',\n"
            "    'validation_mae', 'validation_rmse', 'validation_r2'\n"
            "]])"
        ),
        new_markdown_cell("## K-Means and GMM segmentation with business profiles and PCA"),
        new_code_cell(
            "segmentation = pd.read_csv(REPORTS / 'segmentation_model_selection.csv')\n"
            "profiles = pd.read_csv(REPORTS / 'segment_profiles.csv')\n"
            "pca_sample = pd.read_csv(REPORTS / 'segmentation_pca_sample.csv')\n"
            "display(segmentation)\n"
            "display(profiles)\n"
            "pca_sample.head()"
        ),
        new_markdown_cell("## Next-purchase-category LightGBM and most-popular baseline"),
        new_code_cell(
            "next_category = pd.read_csv(REPORTS / 'next_category_evaluation.csv')\n"
            "display(next_category[[\n"
            "    'model', 'macro_f1', 'top_1_accuracy', 'top_3_accuracy',\n"
            "    'training_rows', 'validation_rows', 'classes'\n"
            "]])"
        ),
        new_markdown_cell("## Item-to-item recommender offline evaluation"),
        new_code_cell(
            "recommender = pd.read_csv(REPORTS / 'recommender_evaluation.csv')\n"
            "display(recommender[[\n"
            "    'model', 'recall_at_5', 'hit_rate_at_5', 'catalog_coverage',\n"
            "    'eligible_evaluation_customers', 'catalog_items'\n"
            "]])"
        ),
        new_markdown_cell("## Local MLflow traceability and artifact reload evidence"),
        new_code_cell(
            "mlflow_runs = pd.read_csv(REPORTS / 'mlflow_run_summary.csv')\n"
            "reload_checks = json.loads((REPORTS / 'artifact_reload_smoke.json').read_text())\n"
            "display(mlflow_runs[['run_id', 'run_name', 'model_family', 'status']])\n"
            "display(pd.DataFrame(reload_checks))"
        ),
        new_markdown_cell(
            "## STEP 04 boundary\n\n"
            "All tuning used training-only five-fold CV and all reported comparison metrics are "
            "validation evidence. Production churn selection, threshold freezing, explainability, "
            "and the one-time final held-out test remain deferred to STEP 06."
        ),
    ]
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return new_notebook(cells=cells, metadata=metadata)


def main() -> int:
    """Write the canonical STEP 04 notebook file."""
    destination = Path("notebooks/03_model_experiments.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_model_experiments_notebook(), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
