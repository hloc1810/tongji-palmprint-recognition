"""Tongji naming convention, image integrity checks, and reproducible within-session folds."""
from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd

EXTENSIONS = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
REQUIRED = {"sample_uid", "session", "palm_id", "path", "fold"}


def palm_id_from_name(stem: str) -> int:
    if not stem.isdecimal() or not (1 <= int(stem) <= 6000):
        raise ValueError(f"Invalid Tongji image stem: {stem!r}")
    return (int(stem) - 1) // 10 + 1


def decode_gray(path: str | Path) -> np.ndarray:
    """Decode an 8-bit ROI using the same OpenCV grayscale rule for every model."""
    encoded = np.fromfile(str(path), dtype=np.uint8)  # Unicode paths on Windows.
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f"ROI must be a readable 8-bit image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.ascontiguousarray(gray)


def read_gray(path: str | Path, image_size: int = 128) -> np.ndarray:
    """Read an Official ROI without resizing or fitting preprocessing on data."""
    gray = decode_gray(path)
    if gray.shape != (image_size, image_size):
        raise ValueError(
            f"Expected Official ROI {image_size}x{image_size}; received {gray.shape} from {path}. "
            "Original hand images require a separately validated ROI extractor."
        )
    return gray


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def build_manifest(root: str | Path, *, strict: bool = True, folds: int = 5,
                   seed: int = 42) -> pd.DataFrame:
    """Return 2-session manifest; sample identifiers are session/stem, never stem alone."""
    root = Path(root).resolve()
    rows = []
    for session in ("session1", "session2"):
        folder = root / session
        if not folder.is_dir():
            raise FileNotFoundError(f"Tongji ROI directory missing: {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            palm = palm_id_from_name(path.stem)
            im = decode_gray(path)
            lap = cv2.Laplacian(im, cv2.CV_64F)
            rows.append({
                "sample_uid": f"{session}/{path.stem}", "session": session,
                "palm_id": palm, "stem": path.stem,
                "path": str(path.relative_to(root).as_posix()),
                "height": int(im.shape[0]), "width": int(im.shape[1]),
                "sha256": _file_sha(path), "brightness": float(im.mean()),
                "contrast": float(im.std()), "blur_proxy": float(lap.var()),
                "fold": -1,
            })
    if not rows:
        raise ValueError("No ROI images found")
    df = pd.DataFrame(rows).sort_values(["session", "palm_id", "stem"]).reset_index(drop=True)
    if df.sample_uid.duplicated().any():
        raise ValueError("Duplicate sample_uid (duplicate stem / multiple extensions)")
    train = df.loc[df.session == "session1"]
    test = df.loc[df.session == "session2"]
    a = train.palm_id.value_counts().sort_index()
    b = test.palm_id.value_counts().sort_index()
    if strict:
        if not ((df.height == 128) & (df.width == 128)).all():
            raise ValueError("Strict Tongji mode expects Official ROI images of size 128x128")
        required = pd.Index(range(1, 601))
        if not a.index.equals(required) or not b.index.equals(required):
            raise ValueError("Strict Tongji mode requires 600 palm IDs in each session")
        if not (a == 10).all() or not (b == 10).all():
            raise ValueError("Strict Tongji mode requires 10 images/palm/session")
    if len(a) != len(b) or set(a.index) != set(b.index):
        raise ValueError("Sessions must contain exactly the same palm IDs")
    if (a < folds).any():
        raise ValueError("Not enough session1 images per class for requested folds")
    rng = np.random.default_rng(seed)
    for palm, sub in train.groupby("palm_id", sort=True):
        indices = sub.index.to_numpy().copy()
        rng.shuffle(indices)
        # 10 images split into 5 folds of 2; each index is held out exactly once.
        for f, chunk in enumerate(np.array_split(indices, folds)):
            df.loc[chunk, "fold"] = f
    validate_manifest(df, folds=folds, strict=strict)
    return df


def validate_manifest(df: pd.DataFrame, *, folds: int, strict: bool) -> None:
    if df.sample_uid.duplicated().any():
        raise ValueError("Duplicate sample_uid")
    if not set(df.session.unique()) == {"session1", "session2"}:
        raise ValueError("Need both sessions")
    a = df[df.session == "session1"]
    b = df[df.session == "session2"]
    if not set(a.palm_id) == set(b.palm_id):
        raise ValueError("Train/test must refer to same closed-set identities")
    if (a.fold < 0).any() or (b.fold != -1).any():
        raise ValueError("Fold labels leak into session2 / missing session1 fold")
    for _, sub in a.groupby("palm_id"):
        if set(sub.fold) != set(range(folds)):
            raise ValueError("A class is absent in some validation fold")
    if strict and (len(a) != 6000 or len(b) != 6000):
        raise ValueError("Strict mode requires 6000 images per session")
    # Identical binary files across different sessions are suspect; do not silently remove them.
    cross = set(a.sha256).intersection(set(b.sha256))
    if cross:
        raise ValueError(f"Cross-session identical file hashes detected: {len(cross)}")


def resolve_path(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    dest = (root / relative).resolve()
    if not dest.is_relative_to(root):
        raise ValueError("Unsafe path outside dataset root")
    return dest


def generate_synthetic_roi(root: str | Path, *, palms: int = 6, samples: int = 4,
                           seed: int = 9) -> None:
    """Explicitly synthetic smoke-test fixture; never valid as Tongji research output."""
    root = Path(root)
    if palms > 600 or samples > 10 or samples < 2:
        raise ValueError("Synthetic fixture must respect Tongji filename block sizes")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128]
    for session_no, session in enumerate(("session1", "session2")):
        folder = root / session
        folder.mkdir(parents=True, exist_ok=True)
        for palm in range(1, palms + 1):
            angle = 0.23 + palm * 0.28
            for j in range(samples):
                phase = (xx * np.cos(angle) + yy * np.sin(angle)) / (2.5 + palm * .22)
                arr = 120 + 65 * np.sin(phase + rng.normal(0, .15))
                arr += rng.normal(0, 7 + session_no * 4, size=(128, 128))
                arr = np.uint8(np.clip(arr, 0, 255))
                stem = f"{(palm-1)*10 + j+1:05d}"
                cv2.imwrite(str(folder / f"{stem}.png"), arr)
