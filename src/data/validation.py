"""Dataset validation: metadata integrity, image files and duplicates.

:func:`validate_dataset` produces a :class:`ValidationReport` describing what
is wrong (``errors``), what deserves attention (``warnings``) and summary
statistics. :func:`select_usable_rows` then keeps only rows that are safe to
train on. Nothing here modifies files on disk.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.image_index import ImageIndex
from src.data.metadata import CLASS_NAMES, REQUIRED_COLUMNS


@dataclass
class ValidationReport:
    """Results of :func:`validate_dataset`. All lists hold ``image_id`` values
    unless stated otherwise, so the report can be saved as JSON."""

    n_rows: int = 0
    n_unique_images: int = 0
    n_unique_lesions: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    unknown_labels: dict[str, int] = field(default_factory=dict)
    duplicate_image_ids: list[str] = field(default_factory=list)
    conflicting_image_labels: list[str] = field(default_factory=list)
    conflicting_image_lesions: list[str] = field(default_factory=list)
    lesions_with_conflicting_labels: list[str] = field(default_factory=list)
    images_per_lesion: dict[int, int] = field(default_factory=dict)
    class_counts_images: dict[str, int] = field(default_factory=dict)
    class_counts_lesions: dict[str, int] = field(default_factory=dict)

    images_checked: bool = False
    n_image_files_found: int = 0
    missing_images: list[str] = field(default_factory=list)
    orphan_images: list[str] = field(default_factory=list)
    ambiguous_images: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    image_sizes: dict[str, int] = field(default_factory=dict)
    #: Groups of byte-identical files: {"image_ids", "lesion_ids", "same_lesion"}.
    duplicate_file_groups: list[dict[str, Any]] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["images_per_lesion"] = {str(k): v for k, v in self.images_per_lesion.items()}
        return d

    @property
    def cross_lesion_duplicates(self) -> list[list[str]]:
        """Image-id groups of identical files that carry different lesion ids.

        These would leak between splits if lesions were the only grouping, so
        :func:`src.data.splitting.build_groups` merges them.
        """
        return [g["image_ids"] for g in self.duplicate_file_groups if not g["same_lesion"]]


def check_metadata(df: pd.DataFrame, report: ValidationReport) -> None:
    """Fill the metadata-only parts of ``report`` (no file access)."""
    report.n_rows = len(df)
    report.n_unique_images = int(df["image_id"].nunique())
    report.n_unique_lesions = int(df["lesion_id"].nunique())

    report.missing_values = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    for col, n in report.missing_values.items():
        if n:
            report.errors.append(f"{n} row(s) have no '{col}'.")

    labels = df["dx"].dropna()
    unknown = labels[~labels.isin(CLASS_NAMES)]
    report.unknown_labels = {str(k): int(v) for k, v in unknown.value_counts().items()}
    if report.unknown_labels:
        report.errors.append(
            f"Unknown dx label(s) {sorted(report.unknown_labels)}; expected {list(CLASS_NAMES)}."
        )

    ids = df["image_id"].dropna()
    dup_ids = sorted(ids[ids.duplicated()].unique().tolist())
    report.duplicate_image_ids = dup_ids
    if dup_ids:
        report.warnings.append(f"{len(dup_ids)} image_id(s) appear in more than one row.")
    labelled = df.dropna(subset=["image_id", "dx"])
    per_image = labelled.groupby("image_id")["dx"].nunique()
    report.conflicting_image_labels = sorted(per_image[per_image > 1].index.tolist())
    if report.conflicting_image_labels:
        report.errors.append(
            f"{len(report.conflicting_image_labels)} image_id(s) have conflicting dx labels."
        )
    per_image_lesion = df.dropna(subset=["image_id", "lesion_id"]).groupby("image_id")["lesion_id"].nunique()
    report.conflicting_image_lesions = sorted(per_image_lesion[per_image_lesion > 1].index.tolist())
    if report.conflicting_image_lesions:
        report.errors.append(
            f"{len(report.conflicting_image_lesions)} image_id(s) are assigned to more than one lesion_id."
        )

    per_lesion_labels = df.dropna(subset=["lesion_id", "dx"]).groupby("lesion_id")["dx"].nunique()
    report.lesions_with_conflicting_labels = sorted(
        per_lesion_labels[per_lesion_labels > 1].index.tolist()
    )
    if report.lesions_with_conflicting_labels:
        report.warnings.append(
            f"{len(report.lesions_with_conflicting_labels)} lesion_id(s) have images with "
            "different dx labels; splitting uses each lesion's most common label."
        )

    unique_pairs = df.dropna(subset=["lesion_id", "image_id"]).drop_duplicates("image_id")
    counts = unique_pairs.groupby("lesion_id").size().value_counts().sort_index()
    report.images_per_lesion = {int(k): int(v) for k, v in counts.items()}

    known = df[df["dx"].isin(CLASS_NAMES)].drop_duplicates("image_id")
    report.class_counts_images = {c: int((known["dx"] == c).sum()) for c in CLASS_NAMES}
    lesion_labels = known.dropna(subset=["lesion_id"]).groupby("lesion_id")["dx"].agg(
        lambda s: s.mode().sort_values().iloc[0]
    )
    report.class_counts_lesions = {c: int((lesion_labels == c).sum()) for c in CLASS_NAMES}
    empty = [c for c, n in report.class_counts_images.items() if n == 0]
    if empty:
        report.warnings.append(f"No images for class(es) {empty}.")


def check_images(
    df: pd.DataFrame,
    index: ImageIndex,
    report: ValidationReport,
    verify_images: bool = True,
    hash_images: bool = True,
) -> None:
    """Fill the file-related parts of ``report``.

    Args:
        verify_images: decode every image fully to detect corrupt files.
        hash_images: compute a SHA-1 of every file to detect byte-identical
            copies stored under different image ids.
    """
    report.images_checked = True
    report.n_image_files_found = len(index)
    ids = df["image_id"].dropna().drop_duplicates()
    known_ids = set(ids)

    report.missing_images = sorted(i for i in ids if i not in index)
    if report.missing_images:
        report.errors.append(f"{len(report.missing_images)} image file(s) listed in metadata were not found.")
    report.orphan_images = sorted(k for k in index.paths if k not in known_ids)
    if report.orphan_images:
        report.warnings.append(f"{len(report.orphan_images)} image file(s) have no metadata row (ignored).")
    report.ambiguous_images = sorted(k for k in index.ambiguous if k in known_ids)
    if report.ambiguous_images:
        report.warnings.append(
            f"{len(report.ambiguous_images)} image_id(s) match several files; the first path "
            "in sorted order is used. Point image_dirs at fewer folders to avoid this."
        )

    lesion_of = df.dropna(subset=["image_id"]).drop_duplicates("image_id").set_index("image_id")["lesion_id"]
    sizes: Counter[str] = Counter()
    by_hash: dict[str, list[str]] = {}
    for image_id in ids:
        path = index.resolve(image_id)
        if path is None:
            continue
        if not (verify_images or hash_images):
            continue
        try:
            data = path.read_bytes()
            if verify_images:
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    sizes[f"{im.width}x{im.height}"] += 1
        except (OSError, UnidentifiedImageError, SyntaxError, ValueError):
            report.unreadable_images.append(image_id)
            continue
        if hash_images:
            by_hash.setdefault(hashlib.sha1(data).hexdigest(), []).append(image_id)

    report.image_sizes = dict(sizes.most_common())
    if report.unreadable_images:
        report.errors.append(f"{len(report.unreadable_images)} image file(s) could not be decoded.")

    for group in by_hash.values():
        if len(group) < 2:
            continue
        lesions = sorted({str(lesion_of.get(i)) for i in group})
        report.duplicate_file_groups.append(
            {"image_ids": sorted(group), "lesion_ids": lesions, "same_lesion": len(lesions) == 1}
        )
    n_cross = len(report.cross_lesion_duplicates)
    if report.duplicate_file_groups:
        report.warnings.append(
            f"{len(report.duplicate_file_groups)} group(s) of byte-identical image files found "
            f"({n_cross} spanning different lesion_ids; these are kept in the same split)."
        )


def validate_dataset(
    df: pd.DataFrame,
    index: ImageIndex | None = None,
    verify_images: bool = True,
    hash_images: bool = True,
) -> ValidationReport:
    """Validate normalised metadata and, if ``index`` is given, the image files."""
    report = ValidationReport()
    check_metadata(df, report)
    if index is not None:
        check_images(df, index, report, verify_images=verify_images, hash_images=hash_images)
    return report


def select_usable_rows(df: pd.DataFrame, report: ValidationReport) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the rows that are safe to use, plus the number dropped per reason.

    A row is dropped if it lacks a required value, has an unknown label, has an
    image id with conflicting labels or lesion ids, or (when images were checked) its file is
    missing or unreadable. Exact repeated rows of the same image keep only the
    first occurrence. Reasons are checked in that order and each row is counted
    once.
    """
    keep = pd.Series(True, index=df.index)
    dropped: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str) -> None:
        newly = mask & keep
        dropped[reason] = int(newly.sum())
        keep.loc[newly] = False

    drop(df[list(REQUIRED_COLUMNS)].isna().any(axis=1), "missing_required_value")
    drop(~df["dx"].isin(CLASS_NAMES), "unknown_label")
    drop(df["image_id"].isin(report.conflicting_image_labels), "conflicting_image_label")
    drop(df["image_id"].isin(report.conflicting_image_lesions), "conflicting_image_lesion")
    if report.images_checked:
        drop(df["image_id"].isin(report.missing_images), "missing_image_file")
        drop(df["image_id"].isin(report.unreadable_images), "unreadable_image")
    drop(df["image_id"].where(keep).duplicated(), "repeated_row")
    return df[keep].reset_index(drop=True), dropped
