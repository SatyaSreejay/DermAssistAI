"""Loading YAML configuration files with portable path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Return ``path`` as an absolute path, interpreting relative paths from ``root``.

    ``~`` is expanded so that home-relative paths work on every OS.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dictionary.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not contain a YAML mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return cfg
