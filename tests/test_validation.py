"""Validation tests: inject known problems into synthetic data and check each is caught."""

import shutil

import pandas as pd
import pytest

from src.data.image_index import build_image_index
from src.data.metadata import CLASS_NAMES, load_metadata
from src.data.validation import select_usable_rows, validate_dataset


def _image_path(ds, image_id):
    return next(ds.image_dir.rglob(f"{image_id}.jpg"))


def test_clean_dataset_has_no_errors(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    report = validate_dataset(df, build_image_index([synthetic.image_dir]))
    assert report.errors == []
    assert report.missing_images == report.unreadable_images == []
    assert sum(report.class_counts_images.values()) == len(df)
    assert list(report.class_counts_images) == list(CLASS_NAMES)
    assert sum(report.class_counts_lesions.values()) == df["lesion_id"].nunique()
    assert sum(k * v for k, v in report.images_per_lesion.items()) == len(df)
    clean, dropped = select_usable_rows(df, report)
    assert len(clean) == len(df) and sum(dropped.values()) == 0


def test_metadata_problems_are_reported_and_dropped(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    df.loc[0, "dx"] = pd.NA  # missing label
    df.loc[1, "dx"] = "melanoma"  # unknown label
    df.loc[2, "lesion_id"] = pd.NA  # missing lesion id
    repeat = df.iloc[[5]].copy()  # same image listed twice, same label
    conflict = df.iloc[[6]].copy()
    conflict["dx"] = "nv" if conflict["dx"].iloc[0] != "nv" else "mel"  # same image, other label
    df = pd.concat([df, repeat, conflict], ignore_index=True)

    report = validate_dataset(df)  # metadata only
    assert report.missing_values == {"lesion_id": 1, "image_id": 0, "dx": 1}
    assert report.unknown_labels == {"melanoma": 1}
    image_5, image_6 = df.loc[5, "image_id"], df.loc[6, "image_id"]
    assert set(report.duplicate_image_ids) == {image_5, image_6}
    assert report.conflicting_image_labels == [image_6]
    assert not report.images_checked

    clean, dropped = select_usable_rows(df, report)
    assert dropped["missing_required_value"] == 2
    assert dropped["unknown_label"] == 1
    assert dropped["conflicting_image_label"] == 2
    assert dropped["repeated_row"] == 1
    assert clean["image_id"].is_unique
    assert image_6 not in set(clean["image_id"])


def test_file_problems_are_reported(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    ids = df["image_id"].tolist()
    _image_path(synthetic, ids[0]).unlink()  # missing image
    _image_path(synthetic, ids[1]).write_bytes(b"not a jpeg")  # corrupt image
    (synthetic.image_dir / "part_1" / "SYN_ORPHAN.jpg").write_bytes(_image_path(synthetic, ids[3]).read_bytes())

    report = validate_dataset(df, build_image_index([synthetic.image_dir]))
    assert report.missing_images == [ids[0]]
    assert report.unreadable_images == [ids[1]]
    assert report.orphan_images == ["SYN_ORPHAN"]
    assert len(report.errors) == 2

    clean, dropped = select_usable_rows(df, report)
    assert dropped["missing_image_file"] == 1 and dropped["unreadable_image"] == 1
    assert len(clean) == len(df) - 2


def test_truncated_image_is_unreadable(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    path = _image_path(synthetic, df["image_id"].iloc[0])
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    report = validate_dataset(df, build_image_index([synthetic.image_dir]), hash_images=False)
    assert report.unreadable_images == [df["image_id"].iloc[0]]


def test_identical_files_across_lesions_are_flagged(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    a = df.iloc[0]
    b = df[df["lesion_id"] != a["lesion_id"]].iloc[0]
    shutil.copyfile(_image_path(synthetic, a["image_id"]), _image_path(synthetic, b["image_id"]))

    report = validate_dataset(df, build_image_index([synthetic.image_dir]))
    assert report.cross_lesion_duplicates == [sorted([a["image_id"], b["image_id"]])]
    assert report.errors == []  # a warning, not an error


def test_report_is_json_serialisable(synthetic):
    import json

    df = load_metadata(synthetic.metadata_csv)
    report = validate_dataset(df, build_image_index([synthetic.image_dir]))
    json.dumps(report.to_dict())


@pytest.mark.parametrize("verify,hash_", [(False, False), (True, False), (False, True)])
def test_optional_checks_can_be_disabled(synthetic, verify, hash_):
    df = load_metadata(synthetic.metadata_csv)
    report = validate_dataset(df, build_image_index([synthetic.image_dir]), verify_images=verify, hash_images=hash_)
    assert report.images_checked
    assert bool(report.image_sizes) == verify
