"""Shared deterministic PyTorch utilities for governed deep-learning models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NumericTransform:
    """Training-fitted median imputation and standardization parameters."""

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the frozen transform to a two-dimensional numeric matrix."""
        matrix = np.asarray(values, dtype="float64")
        filled = np.where(np.isnan(matrix), self.medians, matrix)
        return ((filled - self.means) / self.scales).astype("float32")

    def to_payload(self) -> dict[str, list[float]]:
        """Return JSON/PyTorch-safe transform metadata."""
        return {
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }


def fit_numeric_transform(values: np.ndarray) -> NumericTransform:
    """Fit median imputation and standardization on training values only."""
    matrix = np.asarray(values, dtype="float64")
    medians = np.nanmedian(matrix, axis=0)
    filled = np.where(np.isnan(matrix), medians, matrix)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return NumericTransform(medians=medians, means=means, scales=scales)


def seed_torch(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CPU/GPU behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    """Choose an available PyTorch device without requiring a GPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state to detached CPU tensors for early stopping."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_payload(
    model: nn.Module,
    *,
    model_class: str,
    architecture: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a safe portable PyTorch artifact payload."""
    return {
        "model_class": model_class,
        "architecture": architecture,
        "metadata": metadata,
        "state_dict": clone_state_dict(model),
    }
