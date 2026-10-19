"""Reproducibility and hardware helpers."""

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN choose deterministic kernels.

    Some GPU operations remain slightly non-deterministic, so repeated GPU runs
    can still differ in the last decimals; that is why results are reported
    over several seeds where feasible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preference: str = "auto") -> torch.device:
    """Return ``cuda`` if available (or requested), otherwise ``cpu``."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available. Use --device cpu or auto.")
    return device


def resolve_batch_size(value: int | str, device: torch.device) -> int:
    """``"auto"`` -> 64 on GPU, 16 on CPU; integers are returned unchanged."""
    if value == "auto":
        return 64 if device.type == "cuda" else 16
    return int(value)


def resolve_num_workers(value: int | str) -> int:
    """``"auto"`` -> 2 workers (0 on Windows, where worker processes often cause trouble)."""
    if value == "auto":
        return 0 if platform.system() == "Windows" else min(2, os.cpu_count() or 1)
    return int(value)


def environment_info(device: torch.device) -> dict[str, str]:
    """Versions and hardware, saved with every checkpoint and result."""
    import sklearn
    import torchvision

    info = {
        "python": platform.python_version(),
        # str(): torch.__version__ is a TorchVersion object, which safe checkpoint loading rejects
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    """A DataLoader with seeded shuffling and worker seeds, so runs are repeatable."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
