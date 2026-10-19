"""Class weights for the loss function, computed from training labels only."""

from __future__ import annotations

from typing import Sequence

import numpy as np

WEIGHTING_METHODS = ("none", "balanced", "sqrt")


def class_weights(labels: Sequence[int], n_classes: int, method: str = "balanced") -> np.ndarray | None:
    """Per-class loss weights.

    * ``none``: ``None`` (plain cross-entropy).
    * ``balanced``: ``n / (n_classes * n_c)``, the same formula as
      scikit-learn's ``class_weight="balanced"``. A class with half the
      average frequency gets weight 2.
    * ``sqrt``: square root of ``balanced``, a gentler correction.

    Only pass labels from the training subset: computing weights from
    validation or test labels would leak information about those sets.

    Raises:
        ValueError: if a class has no training examples (its weight would be infinite).
    """
    if method not in WEIGHTING_METHODS:
        raise ValueError(f"Unknown class weighting '{method}'. Choose from {WEIGHTING_METHODS}.")
    if method == "none":
        return None
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    if (counts == 0).any():
        raise ValueError(f"Classes {np.flatnonzero(counts == 0).tolist()} have no training examples.")
    weights = len(labels) / (n_classes * counts)
    return np.sqrt(weights) if method == "sqrt" else weights
