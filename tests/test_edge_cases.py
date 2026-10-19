"""Edge cases from the Phase 2 review: class imbalance, rare classes, inconsistent
labels, and keeping validation/test fixed across training-subset settings."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import make_splits as make_splits_script
from scripts import validate_data
from src.data.image_index import build_image_index
from src.data.metadata import CLASS_NAMES, load_metadata
from src.data.splitting import build_groups, diagnose_splits, make_splits, make_train_subsets
from src.data.synthetic import HAM_LIKE_LESIONS, make_synthetic_dataset
from src.data.validation import select_usable_rows, validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def imbalanced(tmp_path_factory):
    ds = make_synthetic_dataset(tmp_path_factory.mktemp("imbalanced"), lesions_per_class=HAM_LIKE_LESIONS, seed=1)
    return load_metadata(ds.metadata_csv)


# --- class imbalance and rare classes -------------------------------------------------


def test_imbalanced_split_keeps_every_class_in_every_split(imbalanced):
    groups = build_groups(imbalanced)
    splits = make_splits(imbalanced["dx"], groups, 0.15, 0.15, seed=42)
    for name in ("train", "val", "test"):
        assert set(imbalanced["dx"][splits.to_numpy() == name]) == set(CLASS_NAMES), name


def test_imbalanced_split_preserves_class_shares(imbalanced):
    groups = build_groups(imbalanced)
    splits = make_splits(imbalanced["dx"], groups, 0.15, 0.15, seed=42)
    overall = imbalanced["dx"].value_counts(normalize=True)
    for name in ("train", "val", "test"):
        share = imbalanced["dx"][splits.to_numpy() == name].value_counts(normalize=True)
        # the dominant class stays dominant and close to its overall share
        assert share.idxmax() == "nv"
        assert abs(share["nv"] - overall["nv"]) < 0.08, name


def test_rare_class_is_flagged_but_split_still_valid(tmp_path):
    counts = dict.fromkeys(CLASS_NAMES, 60) | {"df": 2}
    ds = make_synthetic_dataset(tmp_path / "rare", lesions_per_class=counts, seed=3)
    df = load_metadata(ds.metadata_csv)
    groups = build_groups(df)
    with pytest.warns(UserWarning, match="least populated class"):  # sklearn's own notice
        splits = make_splits(df["dx"], groups, 0.15, 0.15, seed=42)  # must not crash
    table, warnings = diagnose_splits(df["dx"], groups, splits)
    assert table.loc["df"].sum() == 2
    assert any("'df'" in w for w in warnings)
    assert not any("'nv'" in w for w in warnings)


def test_smallest_subset_keeps_at_least_one_lesion_per_class(imbalanced):
    groups = build_groups(imbalanced)
    subsets = make_train_subsets(imbalanced["dx"], groups, [0.1], seed=0)
    assert set(imbalanced["dx"][subsets["train_10pct"].to_numpy()]) == set(CLASS_NAMES)


def test_subset_rounding_rounds_halves_up():
    labels = pd.Series(["df"] * 5)
    groups = pd.Series([f"L{i}" for i in range(5)])
    subsets = make_train_subsets(labels, groups, [0.5], seed=0)
    assert subsets["train_50pct"].sum() == 3  # 2.5 lesions -> 3, not 2


# --- inconsistent labels and lesion ids ------------------------------------------------


def _lesion_with_several_images(df):
    sizes = df.groupby("lesion_id").size()
    return sizes[sizes >= 2].index[0]


def test_lesion_with_inconsistent_labels_stays_together(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    lesion = _lesion_with_several_images(df)
    rows = df.index[df["lesion_id"] == lesion]
    df.loc[rows[0], "dx"] = "vasc" if df.loc[rows[0], "dx"] != "vasc" else "nv"

    report = validate_dataset(df)
    assert report.lesions_with_conflicting_labels == [lesion]
    assert report.errors == []  # warning only: the rows are kept
    clean, _ = select_usable_rows(df, report)
    assert len(clean) == len(df)

    groups = build_groups(clean)
    splits = make_splits(clean["dx"], groups, 0.15, 0.15, seed=42)
    assert splits[(clean["lesion_id"] == lesion).to_numpy()].nunique() == 1
    subsets = make_train_subsets(clean["dx"], groups, [0.1, 0.5], seed=0)
    for col in subsets.columns:
        assert subsets.loc[(clean["lesion_id"] == lesion).to_numpy(), col].nunique() == 1


def test_image_assigned_to_two_lesions_is_error_and_dropped(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    extra = df.iloc[[0]].copy()
    extra["lesion_id"] = "SYN_OTHER"
    df = pd.concat([df, extra], ignore_index=True)
    report = validate_dataset(df)
    assert report.conflicting_image_lesions == [df.loc[0, "image_id"]]
    clean, dropped = select_usable_rows(df, report)
    assert dropped["conflicting_image_lesion"] == 2
    assert df.loc[0, "image_id"] not in set(clean["image_id"])


# --- validation and test sets stay fixed --------------------------------------------------


def test_val_and_test_identical_for_any_subset_setting(config_file, tmp_path):
    validate_data.main(["--config", str(config_file)])
    outputs = []
    for i, fractions in enumerate([[0.1, 0.25, 0.5, 1.0], [0.5], [0.05, 1.0]]):
        cfg = yaml.safe_load(config_file.read_text())
        cfg["split"]["train_fractions"] = fractions
        cfg["paths"]["processed_dir"] = str(tmp_path / f"processed_{i}")
        variant = tmp_path / f"data_{i}.yaml"
        variant.write_text(yaml.safe_dump(cfg))
        # reuse the same validation outputs
        (tmp_path / f"processed_{i}").mkdir()
        (tmp_path / f"processed_{i}" / "metadata_clean.csv").write_bytes(
            (tmp_path / "processed" / "metadata_clean.csv").read_bytes()
        )
        make_splits_script.main(["--config", str(variant), "--overwrite"])
        outputs.append(pd.read_csv(tmp_path / f"processed_{i}" / "splits.csv"))

    for other in outputs[1:]:
        assert outputs[0]["image_id"].equals(other["image_id"])
        assert outputs[0]["split"].equals(other["split"])
    first = outputs[0]
    subset_cols = [c for c in first.columns if c.startswith("train_")]
    assert not first.loc[first["split"] != "train", subset_cols].to_numpy().any()
    # 10 % subset is identical whether or not other fractions were requested
    assert first["train_100pct"].equals(outputs[2]["train_100pct"])


def test_split_report_lists_rare_class_warnings(config_file, tmp_path):
    validate_data.main(["--config", str(config_file)])
    make_splits_script.main(["--config", str(config_file)])
    report = json.loads((tmp_path / "reports" / "split_report.json").read_text())
    assert set(report["groups_per_class"]) == set(CLASS_NAMES)
    assert isinstance(report["warnings"], list)


# --- Colab notebook consistency --------------------------------------------------------


def _notebook_code():
    nb = json.loads((PROJECT_ROOT / "notebooks" / "colab_quickstart.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_only_uses_existing_script_flags():
    code = _notebook_code()
    calls = re.findall(r'"scripts\.(\w+)"(.*?)\)\s*$', code, flags=re.S | re.M)
    assert len(calls) >= 2, "notebook should call the pipeline scripts"
    for module, rest in calls:
        source = (PROJECT_ROOT / "scripts" / f"{module}.py").read_text(encoding="utf-8")
        for flag in re.findall(r'"(--[a-z][\w-]*)"', rest):
            assert f'"{flag}"' in source, f"{module} has no option {flag}"


def test_notebook_references_existing_files():
    code = _notebook_code()
    for name in re.findall(r'"-r", "(\S+?\.txt)"', code):
        assert (PROJECT_ROOT / name).is_file(), name
    assert "requirements-core.txt" in code
