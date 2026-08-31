"""PyTorch autoencoder for behavioral reconstruction anomaly evidence."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.validation import DataValidationError
from src.models.common import log_mlflow_run
from src.models.torch_common import (
    clone_state_dict,
    fit_numeric_transform,
    seed_torch,
    select_device,
    state_payload,
)


class BehavioralAutoencoder(nn.Module):
    """Compact symmetric 32/8/32 behavioral autoencoder."""

    def __init__(self, input_size: int, hidden_size: int, latent_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct one standardized behavioral vector."""
        return self.decoder(self.encoder(values))


def _reconstruction_loss(
    model: BehavioralAutoencoder,
    loader: DataLoader[tuple[torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for (values,) in loader:
            values = values.to(device)
            loss = torch.mean((model(values) - values) ** 2)
            total += float(loss) * len(values)
            count += len(values)
    return total / max(count, 1)


def train_autoencoder(
    table: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    artifact_path: str | Path,
    mlflow_tags: Mapping[str, str],
) -> dict[str, Any]:
    """Train and evaluate the governed reconstruction anomaly model."""
    required = {"customer_id", "partition", *feature_names}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise DataValidationError(f"Autoencoder table is missing columns: {missing}")
    partition = table["partition"].astype("string")
    train = table.loc[partition.eq("train")]
    validation = table.loc[partition.eq("validation")]
    if train.empty or validation.empty:
        raise DataValidationError("Autoencoder requires non-empty train and validation rows")
    seed = int(metadata["seed"])
    seed_torch(seed)
    transform = fit_numeric_transform(train.loc[:, feature_names].to_numpy(dtype="float64"))
    x_train = transform.transform(train.loc[:, feature_names].to_numpy(dtype="float64"))
    x_validation = transform.transform(validation.loc[:, feature_names].to_numpy(dtype="float64"))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation)),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    device = select_device()
    model = BehavioralAutoencoder(
        len(feature_names), int(config["hidden_size"]), int(config["latent_size"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = clone_state_dict(model)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        running = 0.0
        count = 0
        for (values,) in train_loader:
            values = values.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(values) - values) ** 2)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(values)
            count += len(values)
        train_loss = running / max(count, 1)
        validation_loss = _reconstruction_loss(model, validation_loader, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - float(config["minimum_delta"]):
            best_loss = validation_loss
            best_state = clone_state_dict(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_tensor = torch.from_numpy(x_validation).to(device)
        reconstructed = model(validation_tensor).cpu().numpy()
    squared_errors = (reconstructed - x_validation) ** 2
    row_errors = squared_errors.mean(axis=1)
    threshold_percentile = float(config["threshold_percentile"])
    threshold = float(np.percentile(row_errors, threshold_percentile))
    flags = row_errors >= threshold
    flagged_errors = squared_errors[flags] if flags.any() else squared_errors
    mean_feature_errors = flagged_errors.mean(axis=0)
    contribution_rows = [
        {"feature": name, "mean_flagged_squared_error": float(error)}
        for name, error in sorted(
            zip(feature_names, mean_feature_errors, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    metrics = {
        "train_reconstruction_loss": float(history[-1]["train_loss"]),
        "validation_reconstruction_loss": float(history[-1]["validation_loss"]),
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "validation_error_min": float(row_errors.min()),
        "validation_error_median": float(np.median(row_errors)),
        "validation_error_mean": float(row_errors.mean()),
        "validation_error_max": float(row_errors.max()),
        "flagged_count": int(flags.sum()),
        "flagged_rate": float(flags.mean()),
    }
    parameters = {
        **dict(config),
        "input_size": len(feature_names),
        "best_epoch": best_epoch,
        "device": str(device),
    }
    run_id = log_mlflow_run(
        run_name="behavioral_autoencoder",
        family="autoencoder",
        parameters=parameters,
        metrics={key: float(value) for key, value in metrics.items()},
        tags={**dict(mlflow_tags), "model_name": "behavioral_autoencoder"},
        training_seconds=training_seconds,
    )
    artifact_metadata = {
        **dict(metadata),
        "model_name": "behavioral_autoencoder",
        "feature_names": list(feature_names),
        "transform": transform.to_payload(),
        "metrics": metrics,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "training_seconds": training_seconds,
        "mlflow_run_id": run_id,
        "held_out_test_accessed": False,
        "interpretation": "manual-review anomaly candidate; not confirmed fraud",
    }
    payload = state_payload(
        model,
        model_class="BehavioralAutoencoder",
        architecture={
            "input_size": len(feature_names),
            "hidden_size": int(config["hidden_size"]),
            "latent_size": int(config["latent_size"]),
        },
        metadata=artifact_metadata,
    )
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "metrics": metrics,
        "history": history,
        "feature_contributions": contribution_rows,
        "validation_errors": [
            {
                "customer_id": str(customer_id),
                "reconstruction_error": float(error),
                "is_anomaly_candidate": bool(flag),
            }
            for customer_id, error, flag in zip(
                validation["customer_id"], row_errors, flags, strict=True
            )
        ],
        "artifact": destination.name,
        "metadata": artifact_metadata,
    }


def load_autoencoder(path: str | Path) -> tuple[BehavioralAutoencoder, dict[str, Any]]:
    """Reload a STEP 05 autoencoder artifact onto CPU for inference."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = BehavioralAutoencoder(
        int(architecture["input_size"]),
        int(architecture["hidden_size"]),
        int(architecture["latent_size"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]
