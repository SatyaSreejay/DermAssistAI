"""Leakage-aware train/validation/test splitting.

Why group by lesion? HAM10000 often has several photos of the *same* lesion.
If one photo lands in training and another in the test set, the model is
partly tested on something it has already seen, and scores are inflated. So
every image of a lesion (a *group*) goes to exactly one split.

Why stratify? The classes are very imbalanced (``nv`` dominates). Stratifying
keeps each class's share roughly the same in every split, so rare classes are
present in validation and test. Grouping and stratifying pull against each
other (groups have different sizes), so the resulting fractions are close to,
but not exactly, the requested ones. :func:`summarize_splits` shows how close.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when a group (lesion) appears in more than one split."""


def build_groups(
    df: pd.DataFrame,
    duplicate_image_groups: Iterable[Sequence[str]] = (),
    group_col: str = "lesion_id",
) -> pd.Series:
    """Return a group id per row: the lesion id, with extra merging.

    Byte-identical images that carry *different* lesion ids (see
    ``ValidationReport.cross_lesion_duplicates``) would otherwise leak, so
    their lesions are merged into one group. The group id is the
    alphabetically smallest lesion id in the merged set, which keeps it
    deterministic.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    lesion_of = dict(zip(df["image_id"], df[group_col]))
    for image_ids in duplicate_image_groups:
        lesions = [str(lesion_of[i]) for i in image_ids if i in lesion_of]
        for other in lesions[1:]:
            union(lesions[0], other)
    return df[group_col].astype(str).map(find).rename("group_id")


def group_labels(labels: pd.Series, groups: pd.Series) -> pd.Series:
    """Most common label of each group (ties broken alphabetically)."""
    return labels.groupby(groups).agg(lambda s: s.mode().sort_values().iloc[0])


def _pick_fold(labels: pd.Series, groups: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Return a boolean mask selecting one stratified group fold of about ``fraction``."""
    n_splits = max(2, round(1 / fraction))
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"Only {n_groups} groups available, need at least {n_splits} to hold out "
            f"{fraction:.0%}. Use a larger dataset or a larger fraction."
        )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target = fraction * len(labels)
    folds = [held for _, held in cv.split(np.zeros(len(labels)), labels.to_numpy(), groups.to_numpy())]
    best = min(folds, key=lambda idx: abs(len(idx) - target))
    mask = np.zeros(len(labels), dtype=bool)
    mask[best] = True
    return mask


def make_splits(
    labels: pd.Series,
    groups: pd.Series,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Assign every row to ``train``, ``val`` or ``test``.

    The test set is held out first, then validation is taken from what is left,
    both with :class:`~sklearn.model_selection.StratifiedGroupKFold`. Fractions
    are approximate: a fraction f is realised as one fold out of round(1/f).

    Args:
        labels: class label per row.
        groups: group id per row (from :func:`build_groups`).
        val_fraction, test_fraction: requested fractions of all rows.
        seed: random seed; the same inputs and seed give the same split.
    """
    if not (0 < test_fraction < 1 and 0 < val_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1.")
    labels = labels.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    split = np.full(len(labels), "train", dtype=object)

    test_mask = _pick_fold(labels, groups, test_fraction, seed)
    split[test_mask] = "test"

    rest = np.flatnonzero(~test_mask)
    rel_val = val_fraction / (1 - test_fraction)
    val_mask_rest = _pick_fold(labels.iloc[rest], groups.iloc[rest], rel_val, seed)
    split[rest[val_mask_rest]] = "val"

    result = pd.Series(split, name="split")
    check_no_group_leakage(result, groups)
    return result


def subset_column(fraction: float) -> str:
    """Column name for a training subset, e.g. 0.1 -> ``train_10pct``."""
    return f"train_{round(fraction * 100)}pct"


def make_train_subsets(
    labels: pd.Series,
    groups: pd.Series,
    fractions: Iterable[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Nested, group-aware, stratified subsets of the *training* rows.

    For each class, the training lesions (groups) are shuffled once with
    ``seed``; a fraction f keeps the first ``max(1, floor(f * n + 0.5))`` of them,
    so every class keeps at least one lesion even in the smallest subset.
    Because every fraction takes a prefix of the same order, the 10 % subset
    is contained in the 25 % subset, and so on. Fractions refer to lesions,
    so the share of *images* can differ slightly.

    Returns:
        A DataFrame with one boolean column per fraction, aligned with ``labels``.
    """
    fractions = sorted(set(fractions))
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("Training fractions must be in (0, 1].")
    labels = labels.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    per_group = group_labels(labels, groups)
    rng = np.random.default_rng(seed)

    order: dict[str, list[str]] = {}
    for cls in sorted(per_group.unique()):
        members = sorted(per_group.index[per_group == cls])
        order[cls] = [members[i] for i in rng.permutation(len(members))]

    out = pd.DataFrame(index=labels.index)
    for f in fractions:
        # floor(x + 0.5) rounds halves up (Python's round() would round 2.5 to 2).
        chosen = {g for members in order.values() for g in members[: max(1, math.floor(f * len(members) + 0.5))]}
        out[subset_column(f)] = groups.isin(chosen).to_numpy()
    return out


def check_no_group_leakage(splits: pd.Series, groups: pd.Series) -> None:
    """Raise :class:`LeakageError` if any group appears in more than one split."""
    n_splits_per_group = pd.Series(splits.to_numpy()).groupby(groups.to_numpy()).nunique()
    leaking = n_splits_per_group[n_splits_per_group > 1]
    if len(leaking):
        raise LeakageError(f"{len(leaking)} group(s) appear in several splits, e.g. {list(leaking.index[:5])}")


def diagnose_splits(
    labels: pd.Series, groups: pd.Series, splits: pd.Series, min_groups: int = 5
) -> tuple[pd.DataFrame, list[str]]:
    """Count lesion groups per class and split, and flag weak spots.

    Rare classes have few lesions, so a split may get very few of them (or
    none), which makes per-class metrics on that split unreliable. Each class
    with fewer than ``min_groups`` groups in a split produces a warning.

    Returns:
        (table of group counts with one row per class and one column per split,
        list of warning messages)
    """
    frame = pd.DataFrame(
        {"label": labels.to_numpy(), "group": groups.to_numpy(), "split": splits.to_numpy()}
    )
    # use each group's majority label so a group is counted once
    per_group = group_labels(frame["label"], frame["group"])
    frame["group_label"] = frame["group"].map(per_group)
    table = (
        frame.drop_duplicates("group")
        .pivot_table(index="group_label", columns="split", values="group", aggfunc="count", fill_value=0)
        .reindex(columns=[s for s in SPLIT_NAMES], fill_value=0)
    )
    table.index.name = "dx"
    warnings: list[str] = []
    for cls, row in table.iterrows():
        for split_name in SPLIT_NAMES:
            n = int(row[split_name])
            if n == 0:
                warnings.append(f"Class '{cls}' has no lesions in the {split_name} split.")
            elif n < min_groups:
                warnings.append(
                    f"Class '{cls}' has only {n} lesion(s) in the {split_name} split; "
                    "its per-class metrics there will be unreliable."
                )
    return table, warnings


def summarize_splits(labels: pd.Series, splits: pd.Series) -> pd.DataFrame:
    """Image counts and within-split percentages per class and split.

    Returns a table with one row per class plus a ``TOTAL`` row and columns
    ``<split>_n`` and ``<split>_pct`` for each split and for ``all``.
    """
    table = pd.crosstab(labels.to_numpy(), splits.to_numpy())
    table = table.reindex(columns=[s for s in SPLIT_NAMES if s in table.columns])
    table["all"] = table.sum(axis=1)
    pct = table / table.sum(axis=0) * 100
    out = pd.concat({"n": table, "pct": pct.round(2)}, axis=1)
    out.columns = [f"{split}_{kind}" for kind, split in out.columns]
    total = {f"{s}_n": int(table[s].sum()) for s in table.columns}
    total |= {f"{s}_pct": round(table[s].sum() / table["all"].sum() * 100, 2) for s in table.columns}
    out.loc["TOTAL"] = pd.Series(total)
    out.index.name = "dx"
    ordered = [c for s in table.columns for c in (f"{s}_n", f"{s}_pct")]
    out = out[ordered]
    return out.astype({c: int for c in ordered if c.endswith("_n")})
