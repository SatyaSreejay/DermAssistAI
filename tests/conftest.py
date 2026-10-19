"""Shared pytest fixtures. All data here is synthetic (random noise images)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.data.synthetic import SyntheticDataset, make_synthetic_dataset


@pytest.fixture(scope="session")
def _clean_template(tmp_path_factory) -> SyntheticDataset:
    return make_synthetic_dataset(tmp_path_factory.mktemp("synthetic_template"), lesions_per_class=10, seed=0)


@pytest.fixture
def synthetic(tmp_path, _clean_template) -> SyntheticDataset:
    """A fresh, problem-free copy of the synthetic dataset for each test."""
    root = tmp_path / "synthetic"
    shutil.copytree(_clean_template.root, root)
    return SyntheticDataset(
        root=root,
        metadata_csv=root / _clean_template.metadata_csv.name,
        image_dir=root / "images",
        metadata=_clean_template.metadata.copy(),
    )


@pytest.fixture
def config_file(tmp_path, synthetic) -> Path:
    """A data config pointing at the synthetic dataset and temporary output folders."""
    cfg = {
        "paths": {
            "metadata_csv": str(synthetic.metadata_csv),
            "image_dirs": [str(synthetic.image_dir)],
            "processed_dir": str(tmp_path / "processed"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "validation": {"verify_images": True, "hash_images": True},
        "split": {"seed": 42, "val_fraction": 0.15, "test_fraction": 0.15, "train_fractions": [0.1, 0.25, 0.5, 1.0]},
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path
