"""Create the fixed, lesion-grouped train/val/test split and training subsets.

Run scripts.validate_data first; this script reads its outputs.

Usage (from the project root):
    python -m scripts.make_splits
    python -m scripts.make_splits --seed 7

Outputs (refuses to overwrite unless --overwrite):
    <processed_dir>/splits.csv           image_id, lesion_id, dx, group_id, split, train_XXpct ...
    <reports_dir>/split_summary.csv      class counts and percentages per split
    <reports_dir>/split_report.json      settings, sizes and provenance
    <reports_dir>/split_proportions.png  class shares per split

splits.csv stores ids only, never file paths, so the same split works on any
machine; image paths are resolved at load time from the configured folders.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.plots import plot_split_proportions
from src.data.splitting import (
    SPLIT_NAMES,
    build_groups,
    diagnose_splits,
    make_splits,
    make_train_subsets,
    subset_column,
    summarize_splits,
)
from src.io_utils import check_outputs_free, file_sha256, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    p.add_argument("--processed-dir", help="Folder with metadata_clean.csv; splits.csv is written here too.")
    p.add_argument("--reports-dir", help="Folder with validation_report.json; split reports are written here too.")
    p.add_argument("--seed", type=int, help="Random seed (overrides the config).")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing split.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths, scfg = cfg["paths"], cfg["split"]
    seed = args.seed if args.seed is not None else int(scfg["seed"])
    val_f, test_f = float(scfg["val_fraction"]), float(scfg["test_fraction"])
    fractions = [float(f) for f in scfg["train_fractions"]]

    processed_dir = resolve_path(args.processed_dir or paths["processed_dir"])
    reports_dir = resolve_path(args.reports_dir or paths["reports_dir"])
    clean_csv = processed_dir / "metadata_clean.csv"
    validation_json = reports_dir / "validation_report.json"
    for required in (clean_csv, validation_json):
        if not required.is_file():
            raise FileNotFoundError(f"{required} not found. Run `python -m scripts.validate_data` first.")
    outputs = {
        "splits": processed_dir / "splits.csv",
        "summary": reports_dir / "split_summary.csv",
        "report": reports_dir / "split_report.json",
        "plot": reports_dir / "split_proportions.png",
    }
    check_outputs_free(outputs.values(), args.overwrite)

    df = pd.read_csv(clean_csv, dtype={"lesion_id": str, "image_id": str, "dx": str})
    validation = json.loads(validation_json.read_text(encoding="utf-8"))
    cross_lesion = [g["image_ids"] for g in validation.get("duplicate_file_groups", []) if not g["same_lesion"]]

    groups = build_groups(df, cross_lesion)
    splits = make_splits(df["dx"], groups, val_fraction=val_f, test_fraction=test_f, seed=seed)

    out = df[["image_id", "lesion_id", "dx"]].copy()
    out["group_id"] = groups.to_numpy()
    out["split"] = splits.to_numpy()
    for col in [subset_column(f) for f in fractions]:
        out[col] = False
    train = out["split"] == "train"
    subsets = make_train_subsets(out.loc[train, "dx"], out.loc[train, "group_id"], fractions, seed=seed)
    for col in subsets.columns:
        out.loc[train, col] = subsets[col].to_numpy()

    summary = summarize_splits(out["dx"], out["split"])
    groups_table, split_warnings = diagnose_splits(out["dx"], out["group_id"], out["split"])
    report = {
        "seed": seed,
        "requested": {"val_fraction": val_f, "test_fraction": test_f, "train_fractions": fractions},
        "images": {s: int((out["split"] == s).sum()) for s in SPLIT_NAMES},
        "image_fraction": {s: round(float((out["split"] == s).mean()), 4) for s in SPLIT_NAMES},
        "groups": {s: int(out.loc[out["split"] == s, "group_id"].nunique()) for s in SPLIT_NAMES},
        "groups_per_class": {cls: {k: int(v) for k, v in row.items()} for cls, row in groups_table.iterrows()},
        "warnings": split_warnings,
        "merged_cross_lesion_duplicate_groups": len(cross_lesion),
        "train_subsets": {
            col: {"images": int(out[col].sum()), "groups": int(out.loc[out[col], "group_id"].nunique())}
            for col in subsets.columns
        },
        "images_were_checked": bool(validation.get("images_checked")),
        "provenance": {
            "metadata_clean_sha256": file_sha256(clean_csv),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outputs["splits"], index=False)
    summary.to_csv(outputs["summary"])
    write_json(report, outputs["report"])
    plot_split_proportions(summary, outputs["plot"], "Class share per split (lesion-grouped, stratified)")

    print(summary.to_string())
    print("\nImages per split:", report["images"], " fractions:", report["image_fraction"])
    print("Lesion groups per split:", report["groups"])
    print("Training subsets:", report["train_subsets"])
    for msg in split_warnings:
        print(f"WARNING: {msg}")
    if not report["images_were_checked"]:
        print("NOTE: validation ran with --metadata-only; re-run it where the images are to catch missing files.")
    print("\nWrote", *[f"  {p}" for p in outputs.values()], sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
