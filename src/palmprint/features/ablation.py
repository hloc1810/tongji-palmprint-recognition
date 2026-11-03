"""Feature representation ablation: Raw Grayscale, PCA, and CompCode under 1-NN.

This module implements:
1. Raw grayscale 128x128 pixel vectors (16,384-D), L2-normalized.
2. Randomized PCA (256-D), fit strictly on training folds or session1 (no leakage),
   followed by post-projection L2 normalization.
3. CompCode 600-D orientation histogram baseline.
4. Strictly 1-NN (NearestTemplate) Euclidean distance ranking.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import read_gray, resolve_path
from ..data.preparation import load_manifest
from ..evaluation.metrics import compute_metrics, ranking
from ..models.classical import NearestTemplate
from .pca import PCA


def load_or_extract_raw(
    ws: Path, sessions: tuple[str, ...] = ("session1",)
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load or extract L2-normalized raw grayscale pixel vectors (16,384-D)."""
    ws = Path(ws).resolve()
    full_manifest, info = load_manifest(ws)
    df = full_manifest[full_manifest.session.isin(sessions)].reset_index(drop=True)
    cache_dir = ws / "outputs/feature_ablation/cache"
    cache_path = cache_dir / f"raw_{'_'.join(sessions)}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if np.array_equal(data["uid"], df.sample_uid.to_numpy()):
                return df, data["X"]

    root = Path(info["data_root"])
    features = []
    for path in df.path:
        img = read_gray(resolve_path(root, path), image_size=128)
        vec = img.astype(np.float32) / 255.0
        vec = vec.ravel()
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        features.append(vec)

    X = np.stack(features, axis=0).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, uid=df.sample_uid.to_numpy())
    return df, X


def fit_transform_pca(
    x_train: np.ndarray,
    x_test: np.ndarray | None = None,
    *,
    n_components: int = 256,
    seed: int = 42,
) -> tuple[PCA, np.ndarray, np.ndarray | None]:
    """Fit randomized PCA on x_train, project x_train and optionally x_test, then L2-normalize."""
    x_tr = np.asarray(x_train, dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=seed)
    z_train = pca.fit_transform(x_tr).astype(np.float32)
    norms_tr = np.linalg.norm(z_train, axis=1, keepdims=True)
    z_train = z_train / np.maximum(norms_tr, 1e-12)

    z_test = None
    if x_test is not None:
        x_te = np.asarray(x_test, dtype=np.float32)
        z_test = pca.transform(x_te).astype(np.float32)
        norms_te = np.linalg.norm(z_test, axis=1, keepdims=True)
        z_test = z_test / np.maximum(norms_te, 1e-12)

    return pca, z_train, z_test


def batch_score_1nn(
    model: NearestTemplate, probes: np.ndarray, *, chunk_size: int = 500
) -> np.ndarray:
    """Compute 1-NN squared Euclidean scores in chunks to prevent memory spikes."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    probes = np.asarray(probes, dtype=np.float32)
    scores_list = []
    for start in range(0, len(probes), chunk_size):
        end = min(start + chunk_size, len(probes))
        scores_list.append(model.scores(probes[start:end]))
    return np.vstack(scores_list)


def evaluate_1nn(
    gallery_x: np.ndarray,
    gallery_y: np.ndarray,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
    classes: np.ndarray,
    *,
    chunk_size: int = 500,
) -> tuple[dict, np.ndarray, np.ndarray, float, float]:
    """Fit 1-NN on gallery, evaluate on probe, return metrics, ordered preds, scores, and timings."""
    t0 = time.perf_counter()
    model = NearestTemplate().fit(gallery_x, gallery_y)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = batch_score_1nn(model, probe_x, chunk_size=chunk_size)
    ordered, ordered_scores = ranking(scores, classes)
    infer_time = time.perf_counter() - t1

    metrics = compute_metrics(probe_y, ordered, classes)
    return metrics, ordered, ordered_scores, fit_time, infer_time
