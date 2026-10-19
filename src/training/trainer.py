"""Training loop with staged fine-tuning, early stopping and best-model tracking."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.models.resnet import count_parameters, set_trainable, train_mode


@dataclass
class Stage:
    """One phase of training, e.g. head-only, then fine-tuning layer4."""

    name: str
    trainable: str
    epochs: int
    lr: float
    weight_decay: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stage":
        return cls(d["name"], d["trainable"], int(d["epochs"]), float(d["lr"]), float(d.get("weight_decay", 0.0)))


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    max_batches: int | None = None,
) -> float:
    """One pass over the training data. Returns the mean training loss."""
    train_mode(model)
    total, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = criterion(model(x), y)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    if n == 0:
        raise RuntimeError("The training loader produced no batches.")
    return total / n


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    amp: bool = False,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Class probabilities and true labels for every batch, plus the mean loss.

    The loss uses ``criterion`` as given (with class weights, if any), so it is
    comparable to the training loss.
    """
    model.eval()
    probs, labels = [], []
    total, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
            n += len(y)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    if not probs:
        raise RuntimeError("The evaluation loader produced no batches.")
    return np.concatenate(probs), np.concatenate(labels), (total / n if n else None)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stages: Sequence[Stage],
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    monitor: str = "val_macro_f1",
    patience: int = 3,
    amp: bool = True,
    max_batches: int | None = None,
    on_best: Callable[[dict[str, Any], nn.Module], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train through each stage, selecting the best epoch on the validation set.

    After every epoch the model is evaluated on validation data. Whenever the
    monitored value improves, ``on_best(row, model)`` is called (the training
    script saves a checkpoint there). A stage stops early after ``patience``
    epochs without improvement, and the next stage starts from the best
    weights found so far. The test set is never used here.

    Args:
        monitor: ``"val_macro_f1"`` (higher is better) or ``"val_loss"`` (lower is better).
        max_batches: limit batches per epoch (for quick smoke tests only).

    Returns:
        (history: one dict per epoch, best: the history row of the best epoch)
    """
    if monitor not in ("val_macro_f1", "val_loss"):
        raise ValueError("monitor must be 'val_macro_f1' or 'val_loss'.")
    sign = 1.0 if monitor == "val_macro_f1" else -1.0
    model.to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())

    for stage in stages:
        set_trainable(model, stage.trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=stage.lr, weight_decay=stage.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage.epochs, 1))
        log(f"Stage '{stage.name}': training {stage.trainable} "
            f"({count_parameters(model)['trainable']:,} parameters), up to {stage.epochs} epochs")
        stale = 0
        for epoch in range(1, stage.epochs + 1):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp, max_batches)
            scheduler.step()
            probs, labels, val_loss = predict(model, val_loader, device, criterion, amp, max_batches)
            m = compute_metrics(labels, probs, class_names)
            row = {
                "stage": stage.name,
                "epoch": epoch,
                "lr": lr,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5) if val_loss is not None else None,
                "val_accuracy": round(m["accuracy"], 5),
                "val_balanced_accuracy": round(m["balanced_accuracy"], 5),
                "val_macro_f1": round(m["macro_f1"], 5),
                "seconds": round(time.perf_counter() - start, 2),
            }
            score = sign * row[monitor]
            improved = score > best_score
            row["best"] = improved
            history.append(row)
            log(f"  [{stage.name}] epoch {epoch}: train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
                f"({row['seconds']:.1f}s){'  *best*' if improved else ''}")
            if improved:
                best_score, best, stale = score, dict(row, val_metrics=m), 0
                best_state = copy.deepcopy(model.state_dict())
                if on_best is not None:
                    on_best(best, model)
            else:
                stale += 1
                if stale >= patience:
                    log(f"  early stopping: no improvement in {patience} epochs")
                    break
        model.load_state_dict(best_state)
    return history, best
