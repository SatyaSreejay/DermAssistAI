"""Write a tiny synthetic dataset (random-noise images, NOT medical data)
so the pipeline can be tried end to end without HAM10000.

Usage (from the project root):
    python -m scripts.make_synthetic_dataset --out data/synthetic
"""

from __future__ import annotations

import argparse
import sys

from src.config import resolve_path
from src.data.synthetic import make_synthetic_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic", help="Output folder (must not already hold a dataset).")
    p.add_argument("--lesions-per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ds = make_synthetic_dataset(resolve_path(args.out), lesions_per_class=args.lesions_per_class, seed=args.seed)
    print(f"Wrote {len(ds.metadata)} synthetic images for {ds.metadata['lesion_id'].nunique()} lesions")
    print(f"  metadata: {ds.metadata_csv}\n  images:   {ds.image_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
