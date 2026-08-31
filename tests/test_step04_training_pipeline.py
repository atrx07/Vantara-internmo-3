"""Synthetic end-to-end coverage for the STEP 04 training orchestrator."""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.step04_pipeline import run_step04


def _synthetic_customer_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index in range(120):
        partition = "train" if index < 80 else "validation" if index < 100 else "test"
        frequency = 1 + (index % 12)
        recency = float(5 + (index * 7) % 120)
        net_spend = float(50 + frequency * 25 + (index % 9) * 10)
        average_order = net_spend / frequency
        return_rate = float((index % 5) / 20)
        seasonality = float(0.25 + (index % 4) * 0.1)
        rows.append(
            {
                "customer_id": f"C{index:03d}",
                "cutoff_timestamp": pd.Timestamp("2022-01-01"),
                "frequency_orders": frequency,
                "recency_days": recency,
                "net_spend": net_spend,
                "avg_order_value": average_order,
                "return_rate": return_rate,
                "seasonal_purchase_concentration": seasonality,
                "churn": int((index % 4) in {0, 1}),
                "partition": partition,
                "split_version": "synthetic-split-v1",
            }
        )
    churn = pd.DataFrame(rows)
    clv = churn.drop(columns="churn").copy()
    clv["clv_180d_target"] = (
        clv["frequency_orders"] * 45.0 + clv["net_spend"] * 0.4 - clv["recency_days"] * 0.5
    ).clip(lower=0.0)
    return churn, clv


def _synthetic_transactions(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for index, customer_id in enumerate(customers["customer_id"].astype("string")):
        first_product = f"P{index % 8}"
        second_product = f"P{(index + 2) % 8}"
        records.extend(
            [
                {
                    "customer_id": customer_id,
                    "invoice": f"H{index:03d}",
                    "invoice_date": pd.Timestamp("2021-11-01") + pd.Timedelta(days=index % 20),
                    "stock_code": first_product,
                    "quantity": 1 + index % 3,
                    "gross_positive_value": float(10 + index % 5),
                    "is_positive_purchase": True,
                    "is_product": True,
                },
                {
                    "customer_id": customer_id,
                    "invoice": f"N{index:03d}",
                    "invoice_date": pd.Timestamp("2022-01-10") + pd.Timedelta(days=index % 10),
                    "stock_code": second_product,
                    "quantity": 2 + index % 4,
                    "gross_positive_value": float(20 + index % 7),
                    "is_positive_purchase": True,
                    "is_product": True,
                },
            ]
        )
    taxonomy = pd.DataFrame(
        {
            "stock_code": [f"P{index}" for index in range(8)],
            "category_id": [0, 1, 2, 3, 0, 1, 2, 3],
        }
    )
    return pd.DataFrame(records), taxonomy


def _write_synthetic_project(root: Path) -> Path:
    churn, clv = _synthetic_customer_tables()
    transactions, taxonomy = _synthetic_transactions(churn)
    data_directory = root / "data"
    data_directory.mkdir(parents=True)
    churn.to_parquet(data_directory / "churn.parquet", index=False)
    clv.to_parquet(data_directory / "clv.parquet", index=False)
    transactions.to_parquet(data_directory / "transactions.parquet", index=False)
    taxonomy.to_parquet(data_directory / "taxonomy.parquet", index=False)
    (data_directory / "metadata.json").write_text(
        json.dumps({"source_sha256": "a" * 64}), encoding="utf-8"
    )
    features = [
        "frequency_orders",
        "recency_days",
        "net_spend",
        "avg_order_value",
        "return_rate",
        "seasonal_purchase_concentration",
    ]
    schema_path = root / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-schema-v1",
                "selected_features": features,
                "feature_count": len(features),
            }
        ),
        encoding="utf-8",
    )
    config = {
        "project": {"random_seed": 42},
        "logging": {"level": "INFO"},
        "cleaning": {"interim_transactions": "data/transactions.parquet"},
        "outputs": {
            "churn_features": "data/churn.parquet",
            "clv_features": "data/clv.parquet",
            "product_taxonomy": "data/taxonomy.parquet",
            "step02_metadata": "data/metadata.json",
        },
        "analysis": {"churn_feature_schema": "schema.json"},
        "modeling": {
            "experiment_name": "synthetic-step04",
            "tracking_directory": "mlruns",
            "artifact_directory": "artifacts",
            "evidence_directory": "reports",
            "churn": {
                "logistic_regression": {"penalty": ["l2"], "C": [1.0]},
                "decision_tree": {"candidates": [{"max_depth": 3, "min_samples_leaf": 2}]},
                "random_forest": {
                    "candidates": [
                        {
                            "n_estimators": 8,
                            "max_depth": 4,
                            "min_samples_leaf": 2,
                            "max_features": "sqrt",
                        }
                    ]
                },
                "xgboost": {
                    "candidates": [
                        {
                            "n_estimators": 8,
                            "max_depth": 2,
                            "learning_rate": 0.1,
                            "subsample": 1.0,
                            "colsample_bytree": 1.0,
                        }
                    ]
                },
                "lightgbm": {
                    "candidates": [
                        {
                            "n_estimators": 8,
                            "num_leaves": 7,
                            "learning_rate": 0.1,
                            "min_child_samples": 5,
                        }
                    ]
                },
                "svm": {"candidates": [{"C": 1.0, "gamma": "scale", "kernel": "rbf"}]},
            },
            "clv": {
                "prediction_cap": "training_target_max",
                "ridge_alpha": [1.0],
                "xgboost_candidates": [
                    {
                        "n_estimators": 8,
                        "max_depth": 2,
                        "learning_rate": 0.1,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                    }
                ],
            },
            "segmentation": {
                "features": features,
                "kmeans_candidates": [2],
                "gmm_candidates": [2],
            },
            "next_category": {
                "candidates": [
                    {
                        "n_estimators": 8,
                        "num_leaves": 7,
                        "learning_rate": 0.1,
                        "min_child_samples": 5,
                    }
                ]
            },
            "recommender": {"neighbors": 3, "top_k": 5},
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_synthetic_step04_pipeline_covers_training_logging_and_reload(tmp_path: Path) -> None:
    config_path = _write_synthetic_project(tmp_path)
    summary = run_step04(config_path, project_root=tmp_path)

    assert summary["step"] == "STEP 04"
    assert summary["held_out_test_accessed"] is False
    assert summary["churn_models"] == [
        "decision_tree",
        "lightgbm",
        "logistic_regression",
        "random_forest",
        "svm",
        "xgboost",
    ] or set(summary["churn_models"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm",
    }
    assert summary["mlflow_run_count"] == 12
    assert len(summary["artifact_reload_checks"]) == 11
    assert all(check["reload"] == "PASS" for check in summary["artifact_reload_checks"])
    assert (tmp_path / "reports/step04_summary.json").is_file()
    assert (tmp_path / "artifacts/item_to_item_recommender.joblib").is_file()
