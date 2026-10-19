"""Small helpers for writing output files safely."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def check_outputs_free(paths: Iterable[Path], overwrite: bool) -> None:
    """Refuse to continue if any output already exists, unless ``overwrite``.

    This protects earlier results (for example the fixed data split) from
    being replaced by accident.
    """
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "These output files already exist and would be replaced:\n  "
            + "\n  ".join(existing)
            + "\nMove them away, or re-run with --overwrite if replacing them is intended."
        )


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
