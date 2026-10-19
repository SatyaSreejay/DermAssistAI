from pathlib import Path

import pandas as pd
import pytest

from src.config import resolve_path
from src.data.image_index import build_image_index
from src.data.metadata import MetadataError, load_metadata, normalise_image_id, normalise_metadata


def test_load_metadata_reads_synthetic_file(synthetic):
    df = load_metadata(synthetic.metadata_csv)
    assert len(df) == len(synthetic.metadata)
    assert {"lesion_id", "image_id", "dx"} <= set(df.columns)


def test_missing_required_column_raises():
    with pytest.raises(MetadataError, match="lesion_id"):
        normalise_metadata(pd.DataFrame({"image_id": ["a"], "dx": ["nv"]}))


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata_csv"):
        load_metadata(tmp_path / "nope.csv")


def test_normalisation_strips_whitespace_case_and_extension():
    raw = pd.DataFrame(
        {"lesion_id ": [" HAM_1 ", "HAM_2"], "image_id": ["ISIC_1.JPG ", "ISIC_2"], "dx": [" NV", "  "]}
    )
    df = normalise_metadata(raw)
    assert df.loc[0, "lesion_id"] == "HAM_1"
    assert df.loc[0, "image_id"] == "ISIC_1"
    assert df.loc[0, "dx"] == "nv"
    assert pd.isna(df.loc[1, "dx"])  # blank label becomes missing, not ""


@pytest.mark.parametrize("value,expected", [("ISIC_1.jpg", "ISIC_1"), ("ISIC_1", "ISIC_1"), ("x.png", "x")])
def test_normalise_image_id(value, expected):
    assert normalise_image_id(value) == expected


def test_index_finds_images_across_nested_folders(synthetic):
    index = build_image_index([synthetic.image_dir])
    assert len(index) == len(synthetic.metadata)
    first = synthetic.metadata["image_id"].iloc[0]
    assert index.resolve(first).name == f"{first}.jpg"
    assert index.resolve("does_not_exist") is None


def test_index_is_case_insensitive_and_skips_hidden_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "IMG_1.JPG").write_bytes(b"x")
    (tmp_path / "a" / "._IMG_1.jpg").write_bytes(b"x")  # macOS resource fork
    (tmp_path / "a" / "notes.txt").write_text("hi")
    index = build_image_index([tmp_path])
    assert list(index.paths) == ["IMG_1"]


def test_index_reports_ambiguous_ids(tmp_path):
    for sub in ("part_1", "copy"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "IMG_1.jpg").write_bytes(b"x")
    index = build_image_index([tmp_path])
    assert list(index.ambiguous) == ["IMG_1"]
    assert index.resolve("IMG_1") == sorted(index.paths["IMG_1"])[0]


def test_index_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="image_dirs"):
        build_image_index([tmp_path / "missing"])


def test_resolve_path_relative_and_absolute(tmp_path):
    assert resolve_path("data/raw", root=tmp_path) == (tmp_path / "data" / "raw").resolve()
    assert resolve_path(tmp_path) == tmp_path
    assert isinstance(resolve_path("x"), Path)
