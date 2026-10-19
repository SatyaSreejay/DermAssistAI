"""A tiny synthetic stand-in for HAM10000, used only to test the code.

The images are random coloured noise. They are **not** medical images and
contain no information about skin lesions; models must never be trained or
evaluated on them. Ids are prefixed with ``SYN_`` so they cannot be mistaken
for real ISIC/HAM10000 ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.metadata import CLASS_NAMES


#: Lesion counts per class with roughly the same imbalance as HAM10000
#: (``nv`` dominant, ``df`` and ``vasc`` rare), scaled down for fast tests.
#: These are test settings, not dataset statistics.
HAM_LIKE_LESIONS: dict[str, int] = {"akiec": 12, "bcc": 18, "bkl": 30, "df": 8, "mel": 30, "nv": 200, "vasc": 9}


@dataclass
class SyntheticDataset:
    root: Path
    metadata_csv: Path
    image_dir: Path
    metadata: pd.DataFrame


def make_synthetic_dataset(
    root: str | Path,
    lesions_per_class: int | dict[str, int] = 10,
    max_images_per_lesion: int = 3,
    image_size: tuple[int, int] = (32, 24),
    seed: int = 0,
) -> SyntheticDataset:
    """Write a small HAM10000-shaped dataset under ``root``.

    The layout mirrors a common HAM10000 download: a metadata CSV with the same
    columns and images spread across two "part" folders. Each lesion gets
    between 1 and ``max_images_per_lesion`` images, so the grouping logic is
    exercised. ``lesions_per_class`` is either one number for every class or
    a dict per class (see :data:`HAM_LIKE_LESIONS` for an imbalanced setup).
    ``root`` must not already contain a dataset.
    """
    root = Path(root)
    metadata_csv = root / "SYNTHETIC_metadata.csv"
    if metadata_csv.exists():
        raise FileExistsError(f"{metadata_csv} already exists; choose an empty folder.")
    image_dir = root / "images"
    parts = [image_dir / "part_1", image_dir / "part_2"]
    for p in parts:
        p.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    n_image = 0
    counts = lesions_per_class if isinstance(lesions_per_class, dict) else dict.fromkeys(CLASS_NAMES, lesions_per_class)
    for c, cls in enumerate(CLASS_NAMES):
        for l in range(counts.get(cls, 0)):
            lesion_id = f"SYN_L{c}{l:04d}"
            for _ in range(int(rng.integers(1, max_images_per_lesion + 1))):
                image_id = f"SYN_{n_image:06d}"
                pixels = rng.integers(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
                Image.fromarray(pixels).save(parts[n_image % 2] / f"{image_id}.jpg", quality=90)
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": image_id,
                        "dx": cls,
                        "dx_type": "synthetic",
                        "age": float(rng.integers(20, 80)),
                        "sex": str(rng.choice(["male", "female"])),
                        "localization": "synthetic",
                    }
                )
                n_image += 1
    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_csv, index=False)
    return SyntheticDataset(root, metadata_csv, image_dir, metadata)
