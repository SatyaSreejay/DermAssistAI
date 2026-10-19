"""End-to-end tests of the command-line scripts on synthetic data."""

import json

import pandas as pd
import pytest

from scripts import make_splits, validate_data


def test_validate_then_split_end_to_end(config_file, tmp_path):
    assert validate_data.main(["--config", str(config_file), "--strict"]) == 0
    report = json.loads((tmp_path / "reports" / "validation_report.json").read_text())
    assert report["errors"] == [] and report["images_checked"]
    assert (tmp_path / "reports" / "class_distribution.png").stat().st_size > 0

    assert make_splits.main(["--config", str(config_file)]) == 0
    splits = pd.read_csv(tmp_path / "processed" / "splits.csv")
    assert set(splits["split"]) == {"train", "val", "test"}
    assert splits.groupby("lesion_id")["split"].nunique().max() == 1
    assert not splits.loc[splits["split"] != "train", "train_100pct"].any()
    assert "image_path" not in splits.columns  # ids only, portable across machines
    split_report = json.loads((tmp_path / "reports" / "split_report.json").read_text())
    assert split_report["seed"] == 42


def test_scripts_refuse_to_overwrite(config_file):
    validate_data.main(["--config", str(config_file)])
    with pytest.raises(FileExistsError, match="--overwrite"):
        validate_data.main(["--config", str(config_file)])
    assert validate_data.main(["--config", str(config_file), "--overwrite"]) == 0

    make_splits.main(["--config", str(config_file)])
    with pytest.raises(FileExistsError):
        make_splits.main(["--config", str(config_file)])


def test_metadata_only_mode(config_file, tmp_path):
    assert validate_data.main(["--config", str(config_file), "--metadata-only"]) == 0
    report = json.loads((tmp_path / "reports" / "validation_report.json").read_text())
    assert report["images_checked"] is False


def test_strict_mode_fails_on_errors(config_file, synthetic):
    next(synthetic.image_dir.rglob("*.jpg")).unlink()
    assert validate_data.main(["--config", str(config_file), "--strict"]) == 1


def test_split_requires_validation_first(config_file):
    with pytest.raises(FileNotFoundError, match="validate_data"):
        make_splits.main(["--config", str(config_file)])
