"""Classification metrics shared by every model (CNN, CNN+SVM, CNN+RF).

Why not just accuracy? In HAM10000 most images are ``nv``. A model that
always answers ``nv`` would reach high accuracy while never finding a single
melanoma. Macro-averaged metrics give every class equal weight, so they
expose that failure; per-class recall shows exactly which classes are missed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """All headline metrics for one model on one split.

    Args:
        y_true: true class indices, shape (n,).
        y_prob: predicted class probabilities, shape (n, n_classes). Rows
            should sum to 1. (These are model scores, not calibrated confidences.)
        class_names: names in class-index order.

    Returns a JSON-serialisable dict. ``roc_auc_macro_ovr`` is ``None`` (with
    a reason) when some class has no positive examples in ``y_true``, because
    the one-vs-rest AUC is undefined for that class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)
    if y_prob.ndim != 2 or y_prob.shape != (len(y_true), n_classes):
        raise ValueError(f"y_prob must have shape ({len(y_true)}, {n_classes}), got {y_prob.shape}.")
    if (y_prob < 0).any():
        raise ValueError("y_prob must not contain negative values.")
    # Softmax outputs in float32 can sum to 0.9999999; renormalise in float64.
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    labels = list(range(n_classes))
    y_pred = y_prob.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    result: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-12, 1), labels=labels)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }
    absent = [class_names[i] for i in labels if support[i] == 0]
    if absent:
        result["roc_auc_macro_ovr"] = None
        result["roc_auc_note"] = f"undefined: no examples of {absent} in y_true"
        result["roc_auc_per_class"] = None
    else:
        result["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
        )
        result["roc_auc_per_class"] = {
            name: float(roc_auc_score(y_true == i, y_prob[:, i])) for i, name in enumerate(class_names)
        }
    return result


def per_class_table(metrics: dict[str, Any]):
    """Per-class precision/recall/F1/support (and AUC if defined) as a DataFrame."""
    import pandas as pd

    table = pd.DataFrame(metrics["per_class"]).T
    if metrics.get("roc_auc_per_class"):
        table["roc_auc"] = pd.Series(metrics["roc_auc_per_class"])
    table.index.name = "class"
    return table


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> Path:
    """Heatmap of the confusion matrix (rows: true class, columns: predicted).

    With ``normalize=True`` each row is divided by its total, so the diagonal
    shows per-class recall; raw counts are still printed in every cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(cm, dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    shown = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0) if normalize else counts
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true class" if normalize else "Images")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    threshold = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            label = f"{shown[i, j]:.2f}\n({int(counts[i, j])})" if normalize else f"{int(counts[i, j])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                    color="white" if shown[i, j] > threshold else "#0b0b0b")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
