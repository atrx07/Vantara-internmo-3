"""Grouped rolling-snapshot LSTM for 30-day purchase probability."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.validation import DataValidationError
from src.models.common import classification_metrics, log_mlflow_run
from src.models.torch_common import clone_state_dict, seed_torch, select_device, state_payload


@dataclass(frozen=True)
class RollingSequences:
    """Fixed-width event tensors and snapshot ownership metadata."""

    continuous: np.ndarray
    categories: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    customer_ids: np.ndarray
    partitions: np.ndarray
    cutoffs: np.ndarray


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Torch dataset for padded invoice-event sequences."""

    def __init__(
        self,
        continuous: np.ndarray,
        categories: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.continuous = torch.from_numpy(continuous.astype("float32"))
        self.categories = torch.from_numpy(categories.astype("int64"))
        self.lengths = torch.from_numpy(lengths.astype("int64"))
        self.labels = torch.from_numpy(labels.astype("float32"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.continuous[index],
            self.categories[index],
            self.lengths[index],
            self.labels[index],
        )


class PurchaseLSTM(nn.Module):
    """Single-layer category-embedded LSTM binary classifier."""

    def __init__(
        self,
        category_count: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(category_count, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size + 2, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        continuous: torch.Tensor,
        categories: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return one purchase-within-30-days logit per snapshot."""
        embedded = self.embedding(categories)
        combined = torch.cat([continuous, embedded], dim=2)
        packed = pack_padded_sequence(
            combined, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(1)


def build_rolling_sequences(
    transactions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    split: pd.DataFrame,
    *,
    observation_end: pd.Timestamp,
    config: Mapping[str, Any],
) -> RollingSequences:
    """Build monthly rolling sequences for train/validation customers only."""
    required = {
        "customer_id",
        "invoice",
        "stock_code",
        "invoice_date",
        "quantity",
        "gross_positive_value",
        "is_positive_purchase",
        "is_valid_merchandise",
    }
    missing = sorted(required.difference(transactions.columns))
    if missing:
        raise DataValidationError(f"LSTM transactions are missing columns: {missing}")
    split_required = {"customer_id", "partition"}
    if not split_required.issubset(split.columns):
        raise DataValidationError("LSTM split table is missing customer_id/partition")
    eligible_split = split.loc[
        split["partition"].astype("string").isin(["train", "validation"]),
        ["customer_id", "partition"],
    ].copy()
    partition_by_customer = dict(
        zip(
            eligible_split["customer_id"].astype(str),
            eligible_split["partition"].astype(str),
            strict=True,
        )
    )
    eligible_ids = set(partition_by_customer)
    mask = (
        transactions["customer_id"].notna()
        & transactions["customer_id"].astype("string").isin(eligible_ids)
        & transactions["is_positive_purchase"].fillna(False).astype(bool)
        & transactions["is_valid_merchandise"].fillna(False).astype(bool)
    )
    lines = transactions.loc[
        mask,
        [
            "customer_id",
            "invoice",
            "stock_code",
            "invoice_date",
            "quantity",
            "gross_positive_value",
        ],
    ].copy()
    if lines.empty:
        raise DataValidationError("No eligible train/validation purchase events exist for LSTM")
    category_map = taxonomy.loc[:, ["stock_code", "category_id"]].copy()
    lines = lines.merge(category_map, on="stock_code", how="left", validate="many_to_one")
    lines["category_index"] = lines["category_id"].fillna(-1).astype("int64") + 1
    lines["customer_id"] = lines["customer_id"].astype(str)
    lines["invoice"] = lines["invoice"].astype(str)

    invoice_keys = ["customer_id", "invoice"]
    totals = (
        lines.groupby(invoice_keys, observed=True)
        .agg(invoice_date=("invoice_date", "min"), order_amount=("gross_positive_value", "sum"))
        .reset_index()
    )
    category_totals = (
        lines.groupby([*invoice_keys, "category_index"], observed=True)
        .agg(category_value=("gross_positive_value", "sum"), category_quantity=("quantity", "sum"))
        .reset_index()
        .sort_values(
            [*invoice_keys, "category_value", "category_quantity", "category_index"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(invoice_keys, keep="first")
    )
    events = totals.merge(
        category_totals.loc[:, [*invoice_keys, "category_index"]],
        on=invoice_keys,
        validate="one_to_one",
    ).sort_values(["customer_id", "invoice_date", "invoice"], kind="mergesort")
    events["gap_days"] = (
        events.groupby("customer_id", observed=True)["invoice_date"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    sequence_length = int(config["sequence_length"])
    horizon = pd.Timedelta(days=int(config["horizon_days"]))
    latest_cutoff = pd.Timestamp(observation_end) - horizon
    max_snapshots = int(config["max_snapshots_per_customer"])
    minimum_events = int(config["minimum_history_events"])
    continuous_rows: list[np.ndarray] = []
    category_rows: list[np.ndarray] = []
    lengths: list[int] = []
    labels: list[int] = []
    customers: list[str] = []
    partitions: list[str] = []
    cutoffs: list[np.datetime64] = []
    for customer_id, customer_events in events.groupby("customer_id", sort=True, observed=True):
        customer_events = customer_events.sort_values(["invoice_date", "invoice"], kind="mergesort")
        times = customer_events["invoice_date"].to_numpy(dtype="datetime64[ns]")
        first_cutoff = pd.Timestamp(times[0]).normalize() + pd.offsets.MonthBegin(1)
        candidate_cutoffs = pd.date_range(
            first_cutoff, latest_cutoff, freq=str(config["snapshot_frequency"])
        )
        if max_snapshots > 0:
            candidate_cutoffs = candidate_cutoffs[-max_snapshots:]
        amounts = np.log1p(customer_events["order_amount"].to_numpy(dtype="float64").clip(min=0.0))
        gaps = customer_events["gap_days"].to_numpy(dtype="float64")
        event_categories = customer_events["category_index"].to_numpy(dtype="int64")
        for cutoff in candidate_cutoffs:
            history_end = int(np.searchsorted(times, cutoff.to_datetime64(), side="left"))
            if history_end < minimum_events:
                continue
            history_start = max(0, history_end - sequence_length)
            valid_length = history_end - history_start
            continuous = np.zeros((sequence_length, 2), dtype="float32")
            categories = np.zeros(sequence_length, dtype="int64")
            continuous[:valid_length, 0] = amounts[history_start:history_end]
            continuous[:valid_length, 1] = gaps[history_start:history_end]
            categories[:valid_length] = event_categories[history_start:history_end]
            future_end = int(
                np.searchsorted(times, (cutoff + horizon).to_datetime64(), side="left")
            )
            continuous_rows.append(continuous)
            category_rows.append(categories)
            lengths.append(valid_length)
            labels.append(int(future_end > history_end))
            customers.append(str(customer_id))
            partitions.append(partition_by_customer[str(customer_id)])
            cutoffs.append(cutoff.to_datetime64())
    if not labels:
        raise DataValidationError("Rolling LSTM snapshot generation produced no examples")
    return RollingSequences(
        continuous=np.stack(continuous_rows),
        categories=np.stack(category_rows),
        lengths=np.asarray(lengths, dtype="int64"),
        labels=np.asarray(labels, dtype="int64"),
        customer_ids=np.asarray(customers, dtype=str),
        partitions=np.asarray(partitions, dtype=str),
        cutoffs=np.asarray(cutoffs, dtype="datetime64[ns]"),
    )


def grouped_cv_indices(
    labels: np.ndarray, groups: np.ndarray, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified folds with complete customer isolation."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for train_indices, validation_indices in folds:
        if set(groups[train_indices]).intersection(groups[validation_indices]):
            raise DataValidationError("A customer crossed an LSTM grouped CV fold")
    return folds


def _fit_continuous_scaler(
    continuous: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.concatenate(
        [row[: int(length)] for row, length in zip(continuous, lengths, strict=True)], axis=0
    )
    means = valid.mean(axis=0)
    scales = valid.std(axis=0)
    return means, np.where(scales > 1e-12, scales, 1.0)


def _scale_sequences(
    continuous: np.ndarray, lengths: np.ndarray, means: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    output = continuous.copy().astype("float32")
    for index, length in enumerate(lengths):
        output[index, : int(length)] = (output[index, : int(length)] - means) / scales
    return output


def _make_loader(
    sequences: RollingSequences,
    indices: np.ndarray,
    continuous: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    return DataLoader(
        SequenceDataset(
            continuous[indices],
            sequences.categories[indices],
            sequences.lengths[indices],
            sequences.labels[indices],
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def _evaluate(
    model: PurchaseLSTM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    count = 0
    probabilities: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    with torch.no_grad():
        for continuous, categories, lengths, labels in loader:
            continuous = continuous.to(device)
            categories = categories.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(continuous, categories, lengths)
            loss = criterion(logits, labels)
            total += float(loss) * len(labels)
            count += len(labels)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels_out.append(labels.cpu().numpy())
    return total / max(count, 1), np.concatenate(probabilities), np.concatenate(labels_out)


def _train_one(
    sequences: RollingSequences,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    continuous: np.ndarray,
    category_count: int,
    config: Mapping[str, Any],
    seed: int,
    max_epochs: int,
) -> tuple[PurchaseLSTM, list[dict[str, float | int]], dict[str, Any], float]:
    seed_torch(seed)
    device = select_device()
    model = PurchaseLSTM(
        category_count,
        int(config["embedding_size"]),
        int(config["hidden_size"]),
        float(config["dropout"]),
    ).to(device)
    train_labels = sequences.labels[train_indices]
    positives = max(float(train_labels.sum()), 1.0)
    positive_weight = torch.tensor(
        [(len(train_labels) - positives) / positives], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_loader = _make_loader(
        sequences,
        train_indices,
        continuous,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        sequences,
        validation_indices,
        continuous,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = clone_state_dict(model)
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for values, categories, lengths, labels in train_loader:
            values = values.to(device)
            categories = categories.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values, categories, lengths), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            count += len(labels)
        train_loss = running / max(count, 1)
        validation_loss, _, _ = _evaluate(model, validation_loader, criterion, device)
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
    validation_loss, probabilities, truth = _evaluate(model, validation_loader, criterion, device)
    metrics = classification_metrics(truth.astype("int64"), probabilities)
    metrics["validation_loss"] = validation_loss
    metrics["best_epoch"] = best_epoch
    metrics["positive_weight"] = float(positive_weight.item())
    return model, history, metrics, training_seconds


def train_purchase_lstm(
    sequences: RollingSequences,
    *,
    category_count: int,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    artifact_path: str | Path,
    mlflow_tags: Mapping[str, str],
) -> dict[str, Any]:
    """Run five grouped folds, fit the validation model, and serialize the LSTM."""
    seed = int(metadata["seed"])
    train_indices = np.flatnonzero(sequences.partitions == "train")
    validation_indices = np.flatnonzero(sequences.partitions == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise DataValidationError("LSTM requires non-empty train and validation snapshots")
    folds = grouped_cv_indices(
        sequences.labels[train_indices],
        sequences.customer_ids[train_indices],
        n_splits=int(config["cv_folds"]),
        seed=seed,
    )
    cv_rows: list[dict[str, Any]] = []
    for fold_number, (fold_train, fold_validation) in enumerate(folds, start=1):
        fold_train_indices = train_indices[fold_train]
        fold_validation_indices = train_indices[fold_validation]
        fold_means, fold_scales = _fit_continuous_scaler(
            sequences.continuous[fold_train_indices],
            sequences.lengths[fold_train_indices],
        )
        fold_continuous = _scale_sequences(
            sequences.continuous, sequences.lengths, fold_means, fold_scales
        )
        model, _, metrics, seconds = _train_one(
            sequences,
            fold_train_indices,
            fold_validation_indices,
            continuous=fold_continuous,
            category_count=category_count,
            config=config,
            seed=seed + fold_number,
            max_epochs=int(config["cv_max_epochs"]),
        )
        del model
        run_id = log_mlflow_run(
            run_name=f"purchase_lstm_cv_fold_{fold_number}",
            family="purchase_lstm_cv",
            parameters={**dict(config), "fold": fold_number},
            metrics={key: value for key, value in metrics.items() if isinstance(value, float)},
            tags={**dict(mlflow_tags), "model_name": "purchase_lstm"},
            training_seconds=seconds,
        )
        cv_rows.append(
            {
                "fold": fold_number,
                "train_snapshots": len(fold_train),
                "validation_snapshots": len(fold_validation),
                "train_customers": len(set(sequences.customer_ids[train_indices[fold_train]])),
                "validation_customers": len(
                    set(sequences.customer_ids[train_indices[fold_validation]])
                ),
                **metrics,
                "training_seconds": seconds,
                "mlflow_run_id": run_id,
            }
        )
    means, scales = _fit_continuous_scaler(
        sequences.continuous[train_indices], sequences.lengths[train_indices]
    )
    continuous = _scale_sequences(sequences.continuous, sequences.lengths, means, scales)
    model, history, metrics, training_seconds = _train_one(
        sequences,
        train_indices,
        validation_indices,
        continuous=continuous,
        category_count=category_count,
        config=config,
        seed=seed,
        max_epochs=int(config["max_epochs"]),
    )
    parameters = {
        **dict(config),
        "category_count": category_count,
        "continuous_means": means.tolist(),
        "continuous_scales": scales.tolist(),
    }
    run_id = log_mlflow_run(
        run_name="purchase_lstm_validation",
        family="purchase_lstm",
        parameters=parameters,
        metrics={key: value for key, value in metrics.items() if isinstance(value, float)},
        tags={**dict(mlflow_tags), "model_name": "purchase_lstm"},
        training_seconds=training_seconds,
    )
    artifact_metadata = {
        **dict(metadata),
        "model_name": "purchase_lstm",
        "continuous_features": ["log1p_order_amount", "gap_days"],
        "continuous_means": means.tolist(),
        "continuous_scales": scales.tolist(),
        "train_snapshots": len(train_indices),
        "validation_snapshots": len(validation_indices),
        "train_customers": len(set(sequences.customer_ids[train_indices])),
        "validation_customers": len(set(sequences.customer_ids[validation_indices])),
        "metrics": metrics,
        "parameters": parameters,
        "epochs_ran": len(history),
        "training_seconds": training_seconds,
        "mlflow_run_id": run_id,
        "held_out_test_accessed": False,
    }
    payload = state_payload(
        model,
        model_class="PurchaseLSTM",
        architecture={
            "category_count": category_count,
            "embedding_size": int(config["embedding_size"]),
            "hidden_size": int(config["hidden_size"]),
            "dropout": float(config["dropout"]),
        },
        metadata=artifact_metadata,
    )
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "metrics": metrics,
        "history": history,
        "cv": cv_rows,
        "artifact": destination.name,
        "metadata": artifact_metadata,
    }


def load_purchase_lstm(path: str | Path) -> tuple[PurchaseLSTM, dict[str, Any]]:
    """Reload a STEP 05 LSTM artifact onto CPU for inference."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    architecture = payload["architecture"]
    model = PurchaseLSTM(
        int(architecture["category_count"]),
        int(architecture["embedding_size"]),
        int(architecture["hidden_size"]),
        float(architecture["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["metadata"]
