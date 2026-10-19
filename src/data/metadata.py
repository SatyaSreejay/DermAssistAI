"""Loading and normalising the HAM10000 metadata table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The seven HAM10000 diagnostic categories, in a fixed order used everywhere
#: (class index i always means CLASS_NAMES[i]).
CLASS_NAMES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

CLASS_DESCRIPTIONS: dict[str, str] = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

#: Columns the pipeline cannot work without.
REQUIRED_COLUMNS: tuple[str, ...] = ("lesion_id", "image_id", "dx")

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class MetadataError(ValueError):
    """Raised when the metadata file is unusable (e.g. required columns missing)."""


def normalise_image_id(value: object) -> object:
    """Strip whitespace and a trailing image extension from an image id.

    Some copies of the metadata store ``ISIC_0027419`` and others
    ``ISIC_0027419.jpg``; both map to the file stem ``ISIC_0027419``.
    Missing values are returned unchanged.
    """
    if pd.isna(value):
        return value
    text = str(value).strip()
    for suffix in _IMAGE_SUFFIXES:
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def normalise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a metadata table.

    * column names are stripped of surrounding whitespace;
    * the required columns are converted to strings, stripped, and empty
      strings become missing values (``<NA>``);
    * ``dx`` is lower-cased and ``image_id`` loses any file extension.

    No rows are removed here; deciding which rows are usable is the job of
    :mod:`src.data.validation`.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MetadataError(
            f"Metadata is missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    for col in REQUIRED_COLUMNS:
        series = df[col].astype("string").str.strip()
        df[col] = series.mask(series == "")
    df["dx"] = df["dx"].str.lower()
    df["image_id"] = df["image_id"].map(normalise_image_id).astype("string")
    return df


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read the metadata CSV and normalise it with :func:`normalise_metadata`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        MetadataError: if a required column is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}. Set paths.metadata_csv in "
            "configs/data.yaml or pass --metadata."
        )
    return normalise_metadata(pd.read_csv(path))
