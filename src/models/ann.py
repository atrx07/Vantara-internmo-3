"""PyTorch feed-forward churn ANN on the frozen feature schema."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import classification_metrics, log_mlflow_run, supervised_partitions
from src.models.torch_common import (
    clone_state_dict,
    fit_numeric_transform,
    seed_torch,
    select_device,
    state_payload,
)


class ChurnANN(nn.Module):
    """Governed 128/64 feed-forward binary churn classifier."""

    def __init__(self, input_size: int, dropout_first: float, dropout_second: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_first),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_second),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one churn logit per input row."""
        return self.network(values).squeeze(1)


def _binary_loss(
    model: ChurnANN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            loss = criterion(model(values), labels)
            total += float(loss) * len(labels)
            count += len(labels)
    return total / max(count, 1)


def train_ann(
    table: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    artifact_path: str | Path,
    mlflow_tags: Mapping[str, str],
) -> dict[str, Any]:
    """Train, validate, log, and serialize the governed churn ANN."""
    seed = int(metadata["seed"])
    seed_torch(seed)
    partitions = supervised_partitions(table, feature_names=feature_names, target_name="churn")
    transform = fit_numeric_transform(partitions.x_train.to_numpy(dtype="float64"))
    x_train = transform.transform(partitions.x_train.to_numpy(dtype="float64"))
    x_validation = transform.transform(partitions.x_validation.to_numpy(dtype="float64"))
    y_train = partitions.y_train.to_numpy(dtype="float32")
    y_validation = partitions.y_validation.to_numpy(dtype="float32")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    device = select_device()
    model = ChurnANN(
        len(feature_names), float(config["dropout_first"]), float(config["dropout_second"])
    ).to(device)
    positives = max(float(y_train.sum()), 1.0)
    positive_weight = torch.tensor([(len(y_train) - positives) / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = clone_state_dict(model)
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        running = 0.0
        count = 0
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            count += len(labels)
        train_loss = running / max(count, 1)
        validation_loss = _binary_loss(model, validation_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - float(config["minimum_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_validation).to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    metrics = classification_metrics(y_validation.astype("int64"), probabilities)
    parameters = {
        **dict(config),
        "input_size": len(feature_names),
        "positive_weight": float(positive_weight.item()),
        "best_epoch": best_epoch,
        "device": str(device),
    }
    run_id = log_mlflow_run(
        run_name="ann_churn",
        family="deep_churn",
        parameters=parameters,
        metrics={key: value for key, value in metrics.items() if isinstance(value, float)},
        tags={**dict(mlflow_tags), "model_name": "ann"},
        training_seconds=training_seconds,
    )
    artifact_metadata = {
        **dict(metadata),
        "model_name": "ann",
        "feature_names": list(feature_names),
        "transform": transform.to_payload(),
        "metrics": metrics,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "training_seconds": training_seconds,
        "mlflow_run_id": run_id,
        "held_out_test_accessed": False,
    }
    payload = state_payload(
        model,
        model_class="ChurnANN",
        architecture={
            "input_size": len(feature_names),
            "dropout_first": float(config["dropout_first"]),
            "dropout_second": float(config["dropout_second"]),
        },
        metadata=artifact_metadata,
    )
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "metrics": metrics,
        "history": history,
        "artifact": destination.name,
        "metadata": artifact_metadata,
    }


def load_ann(path: str | Path) -> tuple[ChurnANN, dict[str, Any]]:
    """Reload a STEP 05 ANN artifact onto CPU for inference."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = ChurnANN(
        int(architecture["input_size"]),
        float(architecture["dropout_first"]),
        float(architecture["dropout_second"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]
