"""Mapping ``image_id`` values to image files on disk.

HAM10000 copies come in different layouts (two "part" folders, one combined
folder, extra duplicate folders in the Kaggle archive). Instead of assuming a
layout, we search the configured folders recursively and index every image
file by its file name without extension (its *stem*). The metadata's
``image_id`` (e.g. ``ISIC_0027419``) is then looked up in that index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass
class ImageIndex:
    """All image files found, grouped by file stem."""

    paths: dict[str, list[Path]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, image_id: object) -> bool:
        return image_id in self.paths

    def resolve(self, image_id: str) -> Path | None:
        """Return the file for ``image_id``, or ``None`` if there is none.

        If several files share the stem (for example the same image in two
        folders), the first one in sorted path order is returned so the choice
        is deterministic. :attr:`ambiguous` lists such cases.
        """
        candidates = self.paths.get(image_id)
        return candidates[0] if candidates else None

    @property
    def ambiguous(self) -> dict[str, list[Path]]:
        """Image ids that match more than one file."""
        return {k: v for k, v in self.paths.items() if len(v) > 1}


def build_image_index(
    image_dirs: Iterable[str | Path],
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
) -> ImageIndex:
    """Recursively index image files under each folder in ``image_dirs``.

    Extensions are matched case-insensitively. Hidden files (names starting
    with ``.``, such as macOS ``._ISIC_...`` resource forks) are skipped.

    Raises:
        FileNotFoundError: if a folder does not exist.
    """
    exts = {e.lower() for e in extensions}
    found: dict[str, list[Path]] = {}
    for directory in image_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Image folder not found: {directory}. Check paths.image_dirs in "
                "configs/data.yaml or the --image-dir argument."
            )
        for path in directory.rglob("*"):
            if path.name.startswith(".") or path.suffix.lower() not in exts:
                continue
            if path.is_file():
                found.setdefault(path.stem, []).append(path)
    return ImageIndex({k: sorted(set(v)) for k, v in found.items()})
