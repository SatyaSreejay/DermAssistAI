"""Evaluate a trained checkpoint on the validation or test split.

Use the validation split while developing. Evaluate on the test split only
once, after every modelling decision is final; looking at test results and
then changing the model would make the test score optimistic.

Usage (from the project root):
    python -m scripts.evaluate --checkpoint artifacts/baseline/<run>/best.pt --split val
    python -m scripts.evaluate --checkpoint ... --split test --image-dir /content/ham

Outputs in <checkpoint folder>/eval_<split>/ unless --output-dir is given
(refuses to overwrite unless --overwrite):
    metrics.json               all metrics, timing, model size
    per_class.csv              precision, recall, F1, support (and AUC) per class
    predictions.csv            image_id, true class, predicted class, class probabilities
    confusion_matrix.png       row-normalised, with counts
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.dataset import SkinLesionDataset, load_split_frame
from src.data.image_index import build_image_index
from src.evaluation.metrics import compute_metrics, per_class_table, plot_confusion_matrix
from src.io_utils import check_outputs_free, file_sha256, write_json
from src.models.resnet import count_parameters, model_size_mb
from src.preprocessing.transforms import build_eval_transform
from src.training.checkpoint import load_checkpoint
from src.training.trainer import predict
from src.training.utils import environment_info, get_device, make_loader, resolve_batch_size, resolve_num_workers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"),
                   help="Only used for default data paths.")
    p.add_argument("--splits", help="Path to splits.csv (overrides the config).")
    p.add_argument("--image-dir", action="append", help="Image folder; repeat for several (overrides the config).")
    p.add_argument("--output-dir", help="Where to write results (default: next to the checkpoint).")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--num-workers", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    splits_csv = resolve_path(args.splits or cfg["data"]["splits_csv"])
    image_dirs = [resolve_path(d) for d in (args.image_dir or cfg["data"]["image_dirs"])]
    checkpoint = resolve_path(args.checkpoint)
    out_dir = resolve_path(args.output_dir) if args.output_dir else checkpoint.parent / f"eval_{args.split}"
    outputs = {k: out_dir / v for k, v in {
        "metrics": "metrics.json", "per_class": "per_class.csv",
        "predictions": "predictions.csv", "cm": "confusion_matrix.png",
    }.items()}
    check_outputs_free(outputs.values(), args.overwrite)

    device = get_device(args.device)
    model, meta = load_checkpoint(checkpoint, device)
    class_names = meta["class_names"]
    if meta.get("splits_csv_sha256") and meta["splits_csv_sha256"] != file_sha256(splits_csv):
        print("WARNING: splits.csv differs from the one used in training; results may not be comparable.")
    if args.split == "test":
        print("Evaluating on the TEST split. Do this once, after all model choices are final.")

    frame = load_split_frame(splits_csv, args.split)
    dataset = SkinLesionDataset(frame, build_image_index(image_dirs), build_eval_transform(meta["input"]), class_names)
    loader = make_loader(dataset, resolve_batch_size(args.batch_size, device), False,
                         resolve_num_workers(args.num_workers), 0, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    probs, labels, _ = predict(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    metrics = compute_metrics(labels, probs, class_names)
    metrics |= {
        "split": args.split,
        "checkpoint": str(checkpoint),
        "smoke_checkpoint": bool(meta.get("smoke")),
        "inference": {
            "device": str(device),
            "seconds_total": round(seconds, 3),
            "ms_per_image": round(1000 * seconds / len(labels), 3),
            "note": "includes image loading and preprocessing",
        },
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "environment": environment_info(device),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics, outputs["metrics"])
    per_class_table(metrics).to_csv(outputs["per_class"])
    preds = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
    preds.insert(0, "pred", [class_names[i] for i in probs.argmax(axis=1)])
    preds.insert(0, "true", [class_names[i] for i in labels])
    preds.insert(0, "image_id", dataset.image_ids)
    preds.to_csv(outputs["predictions"], index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, outputs["cm"],
                          title=f"Confusion matrix ({args.split})")

    print(f"{args.split}: n={metrics['n_samples']} accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} balanced_acc={metrics['balanced_accuracy']:.4f} "
          f"weighted_f1={metrics['weighted_f1']:.4f} roc_auc={metrics['roc_auc_macro_ovr']}")
    print(per_class_table(metrics).round(3).to_string())
    if metrics["smoke_checkpoint"]:
        print("NOTE: this checkpoint comes from a smoke run; the numbers are not results.")
    print(f"Inference: {metrics['inference']['ms_per_image']} ms/image on {device}. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
