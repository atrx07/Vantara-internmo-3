"""Structural checks for the required executed STEP 04 experiment notebook."""

from pathlib import Path

import nbformat


def test_model_experiments_notebook_is_executed_thin_consumer() -> None:
    notebook = nbformat.read(Path("notebooks/03_model_experiments.ipynb"), as_version=4)
    markdown = "\n".join(
        cell.source.lower() for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error" for cell in code_cells for output in cell.outputs
    )
    for section in (
        "six classical churn models",
        "180-day customer value",
        "k-means and gmm",
        "next-purchase-category",
        "item-to-item recommender",
        "mlflow",
        "held-out test",
    ):
        assert section in markdown
    assert "src.models.common" in code
    for forbidden in (
        ".fit(",
        "GridSearchCV(",
        "KMeans(",
        "GaussianMixture(",
        "LGBMClassifier(",
        "XGBClassifier(",
    ):
        assert forbidden not in code
