"""Train the ResNet18 baseline (Phase 3).

Uses only the training subset for learning and the validation split for
model selection. The test split is never loaded here; evaluate on it once,
at the end, with scripts.evaluate.

Usage (from the project root):
    # 1. preflight: check config, split and image paths, then stop
    python -m scripts.train_baseline --check-only
    # 2. tiny smoke run (2 batches per epoch, 1 epoch per stage) to test the setup
    python -m scripts.train_baseline --smoke
    # 3. full training (Colab GPU)
    python -m scripts.train_baseline --splits /content/drive/MyDrive/DermAssist-AI/processed/splits.csv \
        --image-dir /content/ham --output-dir /content/drive/MyDrive/DermAssist-AI/artifacts/baseline

Outputs in <output-dir>/<run-name>/ (refuses to overwrite unless --overwrite):
    best.pt                    weights + class names + preprocessing + config + val metrics
    history.csv                one row per epoch (losses, val metrics, time)
    run_config.yaml            the exact settings used
    train_summary.json         data sizes, class weights, timing, model size, environment
    val_metrics.json           full validation metrics of the selected epoch
    val_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch import nn

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data.dataset import SkinLesionDataset, load_split_frame
from src.data.image_index import build_image_index
from src.data.metadata import CLASS_NAMES
from src.evaluation.metrics import plot_confusion_matrix
from src.io_utils import check_outputs_free, file_sha256, write_json
from src.models.resnet import build_resnet18, count_parameters, model_size_mb
from src.preprocessing.transforms import build_eval_transform, build_train_transform
from src.training.checkpoint import save_checkpoint
from src.training.imbalance import class_weights
from src.training.trainer import Stage, fit
from src.training.utils import (
    environment_info,
    get_device,
    make_loader,
    resolve_batch_size,
    resolve_num_workers,
    set_seed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    p.add_argument("--splits", help="Path to splits.csv (overrides the config).")
    p.add_argument("--image-dir", action="append", help="Image folder; repeat for several (overrides the config).")
    p.add_argument("--subset", help="Training subset column, e.g. train_25pct (overrides the config).")
    p.add_argument("--output-dir", help="Parent folder for run folders (overrides the config).")
    p.add_argument("--run-name", help="Run folder name (default: built from model, subset, weighting, seed).")
    p.add_argument("--seed", type=int, help="Random seed (overrides the config).")
    p.add_argument("--class-weighting", choices=["none", "balanced", "sqrt"], help="Overrides the config.")
    p.add_argument("--device", default="auto", help="auto, cuda or cpu.")
    p.add_argument("--batch-size", help="Integer or 'auto' (overrides the config).")
    p.add_argument("--num-workers", help="Integer or 'auto' (overrides the config).")
    p.add_argument("--no-pretrained", action="store_true", help="Random initial weights (offline tests only).")
    p.add_argument("--check-only", action="store_true", help="Validate config, split and image paths, then exit.")
    p.add_argument("--smoke", action="store_true", help="2 batches per epoch, 1 epoch per stage: tests the setup only.")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing an existing run folder's files.")
    return p.parse_args(argv)


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the YAML config with command-line overrides into one resolved dict."""
    cfg = copy.deepcopy(load_config(args.config))
    data, train = cfg["data"], cfg["training"]
    if args.splits:
        data["splits_csv"] = args.splits
    if args.image_dir:
        data["image_dirs"] = args.image_dir
    if args.subset:
        data["train_subset"] = args.subset
    if args.seed is not None:
        train["seed"] = args.seed
    if args.class_weighting:
        train["class_weighting"] = args.class_weighting
    if args.batch_size:
        train["batch_size"] = args.batch_size
    if args.num_workers:
        train["num_workers"] = args.num_workers
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    if args.output_dir:
        cfg["output"]["artifacts_dir"] = args.output_dir
    if args.smoke:
        for stage in train["stages"]:
            stage["epochs"] = 1
    data["splits_csv"] = str(resolve_path(data["splits_csv"]))
    data["image_dirs"] = [str(resolve_path(d)) for d in data["image_dirs"]]
    cfg["output"]["artifacts_dir"] = str(resolve_path(cfg["output"]["artifacts_dir"]))
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_settings(args)
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    seed = int(train_cfg["seed"])
    set_seed(seed)
    device = get_device(args.device)
    class_names = list(CLASS_NAMES)

    run_name = args.run_name or (
        f"{model_cfg['name']}_{data_cfg['train_subset']}_{train_cfg['class_weighting']}_seed{seed}"
        + ("_smoke" if args.smoke else "")
    )
    run_dir = Path(cfg["output"]["artifacts_dir"]) / run_name
    outputs = {k: run_dir / v for k, v in {
        "checkpoint": "best.pt", "history": "history.csv", "config": "run_config.yaml",
        "summary": "train_summary.json", "val_metrics": "val_metrics.json", "cm": "val_confusion_matrix.png",
    }.items()}

    # ---- preflight: everything that can fail cheaply fails before training ----
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    train_frame = load_split_frame(data_cfg["splits_csv"], "train", data_cfg["train_subset"])
    val_frame = load_split_frame(data_cfg["splits_csv"], "val")
    print(f"Indexing images in {data_cfg['image_dirs']} ...")
    index = build_image_index(data_cfg["image_dirs"])
    train_ds = SkinLesionDataset(train_frame, index, build_train_transform(cfg["input"], cfg["augmentation"]), class_names)
    val_ds = SkinLesionDataset(val_frame, index, build_eval_transform(cfg["input"]), class_names)
    missing_classes = sorted(set(class_names) - set(train_frame["dx"]))
    if missing_classes:
        raise ValueError(f"Training subset has no images of {missing_classes}.")
    weights = class_weights(train_ds.labels, len(class_names), train_cfg["class_weighting"])
    batch_size = resolve_batch_size(train_cfg["batch_size"], device)
    num_workers = resolve_num_workers(train_cfg["num_workers"])
    print(f"Train subset '{data_cfg['train_subset']}': {len(train_ds)} images | val: {len(val_ds)} images")
    print("Train images per class:", train_frame["dx"].value_counts().reindex(class_names).to_dict())
    print(f"Class weighting: {train_cfg['class_weighting']}"
          + (f" -> {dict(zip(class_names, weights.round(3).tolist()))}" if weights is not None else ""))
    print(f"Batch size {batch_size}, workers {num_workers}, run folder {run_dir}")
    check_outputs_free(outputs.values(), args.overwrite)
    if args.check_only:
        print("Preflight OK: config, split and all image paths are valid. No training was run.")
        return 0

    # ---- training ----
    model = build_resnet18(len(class_names), pretrained=model_cfg["pretrained"], dropout=model_cfg.get("dropout", 0.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device) if weights is not None else None)
    train_loader = make_loader(train_ds, batch_size, True, num_workers, seed, device)
    val_loader = make_loader(val_ds, batch_size, False, num_workers, seed, device)
    stages = [Stage.from_dict(s) for s in train_cfg["stages"]]
    run_dir.mkdir(parents=True, exist_ok=True)
    with outputs["config"].open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg | {"run_name": run_name, "smoke": args.smoke}, fh, sort_keys=False)

    env = environment_info(device)
    splits_sha = file_sha256(Path(data_cfg["splits_csv"]))

    def on_best(best: dict[str, Any], current: nn.Module) -> None:
        meta = {
            "class_names": class_names,
            "model": model_cfg,
            "input": cfg["input"],
            "training": train_cfg,
            "train_subset": data_cfg["train_subset"],
            "seed": seed,
            "stage": best["stage"],
            "epoch": best["epoch"],
            "val_metrics": {k: best["val_metrics"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
            "splits_csv_sha256": splits_sha,
            "environment": env,
            "smoke": args.smoke,
        }
        save_checkpoint(outputs["checkpoint"], current, meta, overwrite=True)

    start = time.perf_counter()
    history, best = fit(
        model, train_loader, val_loader, stages, criterion, device, class_names,
        monitor=train_cfg.get("monitor", "val_macro_f1"),
        patience=int(train_cfg.get("early_stopping_patience", 3)),
        amp=bool(train_cfg.get("amp", True)),
        max_batches=2 if args.smoke else None,
        on_best=on_best,
    )
    train_seconds = time.perf_counter() - start

    pd.DataFrame(history).to_csv(outputs["history"], index=False)
    val_metrics = best["val_metrics"] | {"split": "val", "stage": best["stage"], "epoch": best["epoch"], "smoke": args.smoke}
    write_json(val_metrics, outputs["val_metrics"])
    plot_confusion_matrix(val_metrics["confusion_matrix"], class_names, outputs["cm"],
                          title=f"Validation confusion matrix ({run_name})")
    write_json({
        "run_name": run_name,
        "smoke": args.smoke,
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "train_class_counts": train_frame["dx"].value_counts().reindex(class_names).fillna(0).astype(int).to_dict(),
        "class_weighting": train_cfg["class_weighting"],
        "class_weights": dict(zip(class_names, weights.tolist())) if weights is not None else None,
        "batch_size": batch_size,
        "epochs_run": len(history),
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "train_seconds_total": round(train_seconds, 1),
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "checkpoint_bytes": outputs["checkpoint"].stat().st_size,
        "splits_csv_sha256": splits_sha,
        "environment": env,
    }, outputs["summary"])

    print(f"\nBest epoch: stage '{best['stage']}' epoch {best['epoch']} "
          f"(val macro-F1 {best['val_macro_f1']:.4f}, val accuracy {best['val_accuracy']:.4f})")
    if args.smoke:
        print("SMOKE RUN: these numbers only show that the pipeline runs; they are not results.")
    print(f"Training time {train_seconds:.1f}s. Outputs in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
