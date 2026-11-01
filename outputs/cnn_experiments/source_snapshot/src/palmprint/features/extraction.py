"""Classical feature extraction and per-session cache validation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage
from skimage.feature import hog

from ..data.dataset import read_gray, resolve_path
from ..data.preparation import load_manifest
from ..utils import hash_file, hash_object, read_yaml
from .compcode import CompCodeConfig, CompCodeExtractor


def feature_config(path: Path) -> CompCodeConfig:
    return CompCodeConfig(**read_yaml(path))


def _feature_for_image(im: np.ndarray, extractor: CompCodeExtractor, rep: str) -> np.ndarray:
    if rep == "compcode":
        return extractor.transform(im)
    if rep == "global":
        return extractor.transform(im, global_histogram=True)
    if rep == "hog":
        z = hog(im, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)
        z /= max(float(np.linalg.norm(z)), 1e-12)
        return z
    raise ValueError(f"Invalid representation {rep}")


def load_features(ws: Path, cfg_path: Path, rep: str = "compcode", *,
                  sessions: tuple[str, ...] = ("session1",)) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load ONLY requested sessions; development never opens session2 ROI pixels.

    Per-session cache keys include the manifest, extractor source SHA and configuration.
    Cached test features cannot exist before an explicit post-lock evaluation request.
    """
    if not sessions or len(set(sessions)) != len(sessions) or not set(sessions) <= {"session1", "session2"}:
        raise ValueError("Invalid feature sessions")
    if "session2" in sessions and not (ws / "artifacts/protocol_lock.json").exists():
        raise RuntimeError("Cannot access session2 features before classical protocol lock")
    full, info = load_manifest(ws)
    df = full[full.session.isin(sessions)].reset_index(drop=True)
    config = feature_config(cfg_path)
    root = Path(info["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    # Check every requested ROI against the audited manifest, including cache hits.
    for row in df.itertuples(index=False):
        if hash_file(resolve_path(root, row.path)) != row.sha256:
            raise RuntimeError(f"Raw ROI changed after dataset audit: {row.sample_uid}")
    source_sha = hash_object({
        "compcode": hash_file(Path(__file__).with_name("compcode.py")),
        "feature_extraction": hash_file(Path(__file__).resolve()),
        "roi_reader": hash_file(Path(__file__).resolve().parents[1] / "data/dataset.py"),
    })
    signature = hash_object({"manifest": info["manifest_sha256"],
                             "config": config.__dict__, "rep": rep,
                             "sessions": sessions, "extractor_source_sha256": source_sha,
                             "libraries": {"numpy": np.__version__, "opencv": cv2.__version__,
                                           "skimage": skimage.__version__},
                             "feature_api_version": "per-session-v3"})
    cache = ws / "outputs/cache" / f"{rep}_{'_'.join(sessions)}_{signature[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            if payload["signature"].item() != signature:
                raise RuntimeError("Feature cache signature mismatch")
            X = payload["X"]
            uid = payload["uid"].tolist()
        dimension = {"compcode": CompCodeExtractor(config).feature_dim,
                     "global": 6,
                     "hog": ((config.image_size // 8 - 1) ** 2) * 2 * 2 * 9}[rep]
        if (uid != df.sample_uid.tolist() or X.shape != (len(df), dimension)
                or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all()):
            raise RuntimeError("Invalid cache shape, sample order or feature values")
        return df, X, signature
    ext = CompCodeExtractor(config)
    X = np.stack([_feature_for_image(read_gray(resolve_path(root, path), config.image_size),
                                     ext, rep) for path in df.path], axis=0).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Invalid feature values")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, uid=np.asarray(df.sample_uid.tolist()), signature=signature)
    return df, X, signature
