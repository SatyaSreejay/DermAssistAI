"""Image transforms for training and evaluation.

Training images are randomly cropped, flipped, rotated and colour-jittered so
the model sees slightly different versions each epoch (data augmentation).
Validation, test and inference images get one fixed, deterministic transform,
so their results are repeatable.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms as T

DEFAULT_INPUT: dict[str, Any] = {
    "resize": 256,
    "image_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def build_eval_transform(input_cfg: dict[str, Any]) -> T.Compose:
    """Resize the shorter side, centre-crop, convert to a normalised tensor."""
    cfg = DEFAULT_INPUT | input_cfg
    return T.Compose(
        [
            T.Resize(cfg["resize"]),
            T.CenterCrop(cfg["image_size"]),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


def build_train_transform(input_cfg: dict[str, Any], aug_cfg: dict[str, Any]) -> T.Compose:
    """Random augmentations followed by the same tensor conversion and normalisation."""
    cfg = DEFAULT_INPUT | input_cfg
    steps: list[Any] = [
        T.Resize(cfg["resize"]),
        T.RandomResizedCrop(cfg["image_size"], scale=tuple(aug_cfg.get("random_resized_crop_scale", (0.8, 1.0)))),
    ]
    if aug_cfg.get("horizontal_flip", True):
        steps.append(T.RandomHorizontalFlip())
    if aug_cfg.get("vertical_flip", True):
        steps.append(T.RandomVerticalFlip())
    if aug_cfg.get("rotation_degrees", 0):
        steps.append(T.RandomRotation(aug_cfg["rotation_degrees"]))
    jitter = aug_cfg.get("color_jitter", 0)
    if jitter:
        steps.append(T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter))
    steps += [T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])]
    return T.Compose(steps)
