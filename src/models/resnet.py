"""ResNet18 classifier with controllable freezing.

Transfer learning in two steps:
1. ``head``: freeze the pretrained backbone and train only the new final
   layer. Fast, and a good baseline when data is limited.
2. ``layer4``: additionally unfreeze the last residual block so the
   high-level features can adapt to dermoscopy images.
"""

from __future__ import annotations

from torch import nn
from torchvision import models

TRAINABLE_STAGES = ("head", "layer4", "layer3", "all")


def build_resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """ResNet18 with its 1000-class ImageNet layer replaced by ``num_classes`` outputs.

    ``pretrained=True`` downloads the torchvision ImageNet weights (about 45 MB)
    the first time; tests use ``pretrained=False`` to stay offline.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features  # 512
    head: list[nn.Module] = [nn.Dropout(dropout)] if dropout > 0 else []
    head.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*head)
    return model


def set_trainable(model: nn.Module, stage: str) -> None:
    """Freeze everything, then unfreeze the parts named by ``stage``.

    ``head`` -> fc only; ``layer4`` -> layer4 + fc; ``layer3`` -> layer3,
    layer4 + fc; ``all`` -> every parameter.
    """
    if stage not in TRAINABLE_STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {TRAINABLE_STAGES}.")
    for p in model.parameters():
        p.requires_grad = stage == "all"
    unfreeze = {"head": ["fc"], "layer4": ["layer4", "fc"], "layer3": ["layer3", "layer4", "fc"], "all": []}[stage]
    for name in unfreeze:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def train_mode(model: nn.Module) -> None:
    """Put the model in training mode, but keep frozen BatchNorm layers in eval mode.

    A frozen BatchNorm layer should keep its pretrained running statistics;
    in plain ``train()`` mode it would keep updating them from our batches.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: nn.Module) -> float:
    """Size of the parameters and buffers in megabytes (float32 on disk)."""
    n_bytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    return round(n_bytes / 2**20, 2)

