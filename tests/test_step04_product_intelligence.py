"""STEP 04 segmentation, next-category, and recommender unit tests."""

import pandas as pd

from src.models.next_category import build_next_category_targets
from src.recommendation.item_to_item import _interaction_matrix, _leave_last_order_out


def test_next_category_uses_next_invoice_and_locked_tie_breaks() -> None:
    transactions = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B", "B"],
            "invoice": ["I1", "I1", "I2", "J1", "J1"],
            "invoice_date": pd.to_datetime(
                ["2022-02-01", "2022-02-01", "2022-03-01", "2022-02-02", "2022-02-02"]
            ),
            "stock_code": ["P1", "P2", "P3", "P1", "PX"],
            "quantity": [1, 3, 100, 2, 1],
            "gross_positive_value": [10.0, 10.0, 1000.0, 8.0, 20.0],
            "is_positive_purchase": [True] * 5,
            "is_product": [True] * 5,
        }
    )
    taxonomy = pd.DataFrame({"stock_code": ["P1", "P2", "P3"], "category_id": [2, 4, 5]})
    targets = build_next_category_targets(
        transactions,
        taxonomy,
        cutoff=pd.Timestamp("2022-01-31"),
    ).set_index("customer_id")

    assert int(targets.loc["A", "next_category_id"]) == 4
    assert int(targets.loc["B", "next_category_id"]) == -1


def test_recommender_leave_last_order_out_and_log_quantity_matrix() -> None:
    lines = pd.DataFrame(
        {
            "customer_id": ["A", "A", "A", "B"],
            "invoice": ["I1", "I1", "I2", "J1"],
            "invoice_date": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-02-01", "2022-01-03"]
            ),
            "stock_code": ["P1", "P2", "P3", "P1"],
            "quantity": [1, 2, 3, 4],
        }
    )
    history, holdout = _leave_last_order_out(lines)
    assert set(history["stock_code"]) == {"P1", "P2"}
    assert set(holdout["stock_code"]) == {"P3"}
    matrix, customers, items, _, _ = _interaction_matrix(history)
    assert customers == ["A"]
    assert items == ["P1", "P2"]
    assert matrix.shape == (1, 2)
    assert matrix.data.min() > 0
