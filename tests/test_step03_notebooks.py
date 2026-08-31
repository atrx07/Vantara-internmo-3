"""Structural checks for required executed STEP 03 notebooks."""

from pathlib import Path

import nbformat


def _notebook(path: str) -> nbformat.NotebookNode:
    return nbformat.read(Path(path), as_version=4)


def test_eda_notebook_is_executed_consumer_with_required_sections() -> None:
    notebook = _notebook("notebooks/01_eda.ipynb")
    markdown = "\n".join(
        cell.source.lower() for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error" for cell in code_cells for output in cell.outputs
    )
    assert "src.analysis.eda" in code
    for required in (
        "spend",
        "frequency",
        "recency",
        "churn",
        "country",
        "seasonality",
        "outlier",
        "hypotheses",
    ):
        assert required in markdown


def test_feature_notebook_is_executed_consumer_with_correlation_vif_and_freeze() -> None:
    notebook = _notebook("notebooks/02_feature_engineering.ipynb")
    markdown = "\n".join(
        cell.source.lower() for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error" for cell in code_cells for output in cell.outputs
    )
    assert "src.analysis.eda" in code
    assert "correlation" in markdown
    assert "vif" in markdown
    assert "frozen churn model feature order" in markdown
    for notebook_only_pattern in (".groupby(", "train_test_split(", "TfidfVectorizer("):
        assert notebook_only_pattern not in code
