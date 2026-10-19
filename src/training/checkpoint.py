"""Saving and loading model checkpoints together with everything needed to reuse them.

A checkpoint stores the weights *and* the class order, preprocessing settings,
model settings, training settings, seed, validation metrics and software
versions, so a model can be evaluated or served later without guessing how it
was made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.resnet import build_resnet18

CHECKPOINT_FORMAT = 1


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict[str, Any], overwrite: bool = False) -> Path:
    """Write ``{"state_dict", "meta", "format"}`` to ``path``.

    ``meta`` must contain ``class_names``, ``model`` (its config) and ``input``
    (preprocessing config). Existing files are only replaced with ``overwrite=True``.
    """
    path = Path(path)
    for key in ("class_names", "model", "input"):
        if key not in meta:
            raise ValueError(f"Checkpoint meta must include '{key}'.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"format": CHECKPOINT_FORMAT, "state_dict": state, "meta": meta}, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model described in the checkpoint and load its weights.

    The model is built without downloading ImageNet weights, because the
    checkpoint already holds the trained ones. Returns ``(model_in_eval_mode, meta)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=True refuses arbitrary pickled code; meta holds only plain data.
    payload = torch.load(path, map_location=device, weights_only=True)
    meta = payload["meta"]
    model_cfg = meta["model"]
    if model_cfg.get("name", "resnet18") != "resnet18":
        raise ValueError(f"Unsupported model '{model_cfg.get('name')}' in {path}.")
    model = build_resnet18(len(meta["class_names"]), pretrained=False, dropout=model_cfg.get("dropout", 0.0))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, meta
