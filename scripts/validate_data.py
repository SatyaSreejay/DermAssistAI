"""Validate the HAM10000 metadata and image files.

Usage (from the project root):
    python -m scripts.validate_data                       # paths from configs/data.yaml
    python -m scripts.validate_data --metadata-only       # no images available yet
    python -m scripts.validate_data --image-dir /content/drive/MyDrive/HAM10000

Outputs (refuses to overwrite unless --overwrite):
    <reports_dir>/validation_report.json   everything that was found
    <reports_dir>/class_distribution.png   images per class
    <processed_dir>/metadata_clean.csv     rows that passed validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.image_index import build_image_index
from src.data.metadata import load_metadata
from src.data.plots import plot_class_distribution
from src.data.validation import select_usable_rows, validate_dataset
from src.io_utils import check_outputs_free, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    p.add_argument("--metadata", help="Path to the metadata CSV (overrides the config).")
    p.add_argument("--image-dir", action="append", help="Image folder; repeat for several (overrides the config).")
    p.add_argument("--processed-dir", help="Where to write metadata_clean.csv (overrides the config).")
    p.add_argument("--reports-dir", help="Where to write the report and plot (overrides the config).")
    p.add_argument("--metadata-only", action="store_true", help="Skip all image-file checks.")
    p.add_argument("--no-verify", action="store_true", help="Do not decode images (faster).")
    p.add_argument("--no-hash", action="store_true", help="Do not hash images for duplicate detection.")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs.")
    p.add_argument("--strict", action="store_true", help="Exit with code 1 if any error is found.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths, vcfg = cfg["paths"], cfg.get("validation", {})

    metadata_csv = resolve_path(args.metadata or paths["metadata_csv"])
    processed_dir = resolve_path(args.processed_dir or paths["processed_dir"])
    reports_dir = resolve_path(args.reports_dir or paths["reports_dir"])
    out_report = reports_dir / "validation_report.json"
    out_plot = reports_dir / "class_distribution.png"
    out_clean = processed_dir / "metadata_clean.csv"
    check_outputs_free([out_report, out_plot, out_clean], args.overwrite)

    print(f"Metadata: {metadata_csv}")
    df = load_metadata(metadata_csv)

    index = None
    if not args.metadata_only:
        image_dirs = [resolve_path(d) for d in (args.image_dir or paths["image_dirs"])]
        print("Indexing images in:", *[f"  {d}" for d in image_dirs], sep="\n")
        index = build_image_index(image_dirs)
        print(f"  found {len(index)} distinct image file names")

    report = validate_dataset(
        df,
        index,
        verify_images=vcfg.get("verify_images", True) and not args.no_verify,
        hash_images=vcfg.get("hash_images", True) and not args.no_hash,
    )
    clean, dropped = select_usable_rows(df, report)

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    result = report.to_dict() | {
        "metadata_csv": str(metadata_csv),
        "image_dirs": [str(d) for d in (args.image_dir or paths["image_dirs"])] if index is not None else [],
        "rows_kept": len(clean),
        "rows_dropped": dropped,
    }
    write_json(result, out_report)
    clean.to_csv(out_clean, index=False)
    plot_class_distribution(report.class_counts_images, out_plot, "Images per class (usable metadata rows)")

    print(f"\nRows: {report.n_rows}  images: {report.n_unique_images}  lesions: {report.n_unique_lesions}")
    print("Images per class:", report.class_counts_images)
    print("Lesions per class:", report.class_counts_lesions)
    print("Images per lesion (n_images: n_lesions):", report.images_per_lesion)
    if not report.images_checked:
        print("Image files were NOT checked (--metadata-only).")
    for msg in report.errors:
        print(f"ERROR:   {msg}")
    for msg in report.warnings:
        print(f"WARNING: {msg}")
    print(f"Kept {len(clean)} of {report.n_rows} rows; dropped {dropped}")
    print(f"\nWrote {out_report}\n      {out_plot}\n      {out_clean}")
    return 1 if (args.strict and report.errors) else 0


if __name__ == "__main__":
    sys.exit(main())
