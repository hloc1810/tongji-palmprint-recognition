"""Small deterministic, auditable IO utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def hash_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def hash_object(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def atomic_json(path: str | Path, value: dict | list) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dst.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        tmp = Path(f.name)
    os.replace(tmp, dst)


def env_info() -> dict:
    import cv2
    import pandas
    import scipy
    import skimage
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "opencv": cv2.__version__,
        "pandas": pandas.__version__,
        "skimage": skimage.__version__,
    }
