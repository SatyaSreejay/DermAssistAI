"""PyTorch dataset reading images listed in ``splits.csv``."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.image_index import ImageIndex
from src.data.metadata import CLASS_NAMES

SPLIT_COLUMNS = ("image_id", "lesion_id", "dx", "split")


class MissingImagesError(FileNotFoundError):
    """Raised when images listed in the split cannot be found."""


def load_split_frame(splits_csv: str | Path, split: str, subset: str | None = None) -> pd.DataFrame:
    """Rows of ``splits.csv`` for one split, optionally restricted to a training subset.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        subset: a subset column such as ``"train_25pct"`` (training split only).
    """
    splits_csv = Path(splits_csv)
    if not splits_csv.is_file():
        raise FileNotFoundError(f"{splits_csv} not found. Run `python -m scripts.make_splits` first.")
    df = pd.read_csv(splits_csv, dtype={"image_id": str, "lesion_id": str, "dx": str})
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{splits_csv} is missing columns {missing}.")
    rows = df[df["split"] == split]
    if subset is not None:
        if split != "train":
            raise ValueError("Training subsets only apply to the 'train' split.")
        if subset not in df.columns:
            options = [c for c in df.columns if c.startswith("train_")]
            raise ValueError(f"Unknown subset '{subset}'. Available: {options}")
        rows = rows[rows[subset].astype(bool)]
    if rows.empty:
        raise ValueError(f"No rows for split={split!r}, subset={subset!r} in {splits_csv}.")
    return rows.reset_index(drop=True)


class SkinLesionDataset(Dataset):
    """Returns ``(image_tensor, class_index)`` pairs.

    All image paths are resolved when the dataset is created, so a missing
    file stops the run immediately instead of crashing halfway through training.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        index: ImageIndex,
        transform: Callable[[Image.Image], torch.Tensor],
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.class_names = list(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        unknown = sorted(set(frame["dx"]) - set(self.class_names))
        if unknown:
            raise ValueError(f"Labels {unknown} are not in class_names {self.class_names}.")
        paths = [index.resolve(i) for i in frame["image_id"]]
        missing = [i for i, p in zip(frame["image_id"], paths) if p is None]
        if missing:
            raise MissingImagesError(
                f"{len(missing)} image(s) listed in the split were not found, e.g. {missing[:10]}. "
                "Check --image-dir, or re-run validation and the split on this copy of the data."
            )
        self.image_ids = frame["image_id"].tolist()
        self.paths: list[Path] = paths  # type: ignore[assignment]
        self.labels = [self.class_to_idx[d] for d in frame["dx"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[i]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.labels[i]
