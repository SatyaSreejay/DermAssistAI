"""Splitting tests: no lesion leakage, stratification, determinism, nested subsets."""

import pandas as pd
import pytest

from src.data.metadata import CLASS_NAMES, load_metadata
from src.data.splitting import (
    LeakageError,
    build_groups,
    check_no_group_leakage,
    make_splits,
    make_train_subsets,
    summarize_splits,
)


@pytest.fixture
def df(synthetic):
    return load_metadata(synthetic.metadata_csv)


def _split(df, seed=42, dup_groups=()):
    groups = build_groups(df, dup_groups)
    return groups, make_splits(df["dx"], groups, 0.15, 0.15, seed=seed)


def test_every_row_gets_exactly_one_split(df):
    _, splits = _split(df)
    assert len(splits) == len(df)
    assert set(splits) == {"train", "val", "test"}


def test_no_lesion_appears_in_two_splits(df):
    _, splits = _split(df)
    per_lesion = pd.Series(splits.to_numpy()).groupby(df["lesion_id"].to_numpy()).nunique()
    assert (per_lesion == 1).all()


def test_split_sizes_are_close_to_requested(df):
    _, splits = _split(df)
    frac = splits.value_counts(normalize=True)
    assert 0.08 <= frac["test"] <= 0.22
    assert 0.08 <= frac["val"] <= 0.22


def test_every_class_present_in_every_split(df):
    _, splits = _split(df)
    for name in ("train", "val", "test"):
        assert set(df["dx"][splits.to_numpy() == name]) == set(CLASS_NAMES)


def test_same_seed_same_split_different_seed_differs(df):
    _, a = _split(df, seed=1)
    _, b = _split(df, seed=1)
    _, c = _split(df, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_identical_images_from_different_lesions_stay_together(df):
    # Pair one image from each of several lesions, then check each pair shares a split.
    lesions = df.drop_duplicates("lesion_id").reset_index(drop=True)
    pairs = [[lesions.loc[i, "image_id"], lesions.loc[i + 1, "image_id"]] for i in range(0, 20, 2)]
    groups, splits = _split(df, dup_groups=pairs)
    split_of = dict(zip(df["image_id"], splits))
    for a, b in pairs:
        assert split_of[a] == split_of[b]
        # all images of both lesions end up in one group
        la, lb = df.set_index("image_id").loc[[a, b], "lesion_id"]
        assert groups[df["lesion_id"] == la].iloc[0] == groups[df["lesion_id"] == lb].iloc[0]


def test_leakage_check_raises():
    with pytest.raises(LeakageError):
        check_no_group_leakage(pd.Series(["train", "test"]), pd.Series(["L1", "L1"]))


def test_invalid_fractions_raise(df):
    groups = build_groups(df)
    with pytest.raises(ValueError):
        make_splits(df["dx"], groups, val_fraction=0.6, test_fraction=0.5)


def test_too_few_groups_raises():
    labels = pd.Series(["nv", "mel", "nv"])
    with pytest.raises(ValueError, match="groups"):
        make_splits(labels, pd.Series(["a", "b", "c"]), 0.15, 0.15)


def test_train_subsets_are_nested_grouped_and_cover_all_classes(df):
    groups, splits = _split(df)
    train = splits.to_numpy() == "train"
    labels, tgroups = df["dx"][train], groups[train]
    subsets = make_train_subsets(labels, tgroups, [0.1, 0.25, 0.5, 1.0], seed=0)

    assert list(subsets.columns) == ["train_10pct", "train_25pct", "train_50pct", "train_100pct"]
    assert subsets["train_100pct"].all()
    for small, big in [("train_10pct", "train_25pct"), ("train_25pct", "train_50pct"), ("train_50pct", "train_100pct")]:
        assert not (subsets[small] & ~subsets[big]).any(), f"{small} not inside {big}"
    for col in subsets.columns:
        chosen = subsets[col].to_numpy()
        # a lesion is either fully in or fully out of a subset
        per_group = pd.Series(chosen).groupby(tgroups.to_numpy()).nunique()
        assert (per_group == 1).all()
        assert set(labels[chosen]) == set(CLASS_NAMES)  # at least one lesion per class


def test_subsets_are_deterministic(df):
    groups = build_groups(df)
    a = make_train_subsets(df["dx"], groups, [0.25], seed=3)
    b = make_train_subsets(df["dx"], groups, [0.25], seed=3)
    assert a.equals(b)


def test_summary_counts_add_up(df):
    _, splits = _split(df)
    summary = summarize_splits(df["dx"], splits)
    assert summary.loc["TOTAL", "all_n"] == len(df)
    assert summary.loc[list(CLASS_NAMES), "train_pct"].sum() == pytest.approx(100, abs=0.1)
