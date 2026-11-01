"""Metrics, deterministic ranking, and cluster bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def ranking(scores: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All methods supply *smaller is better* scores and globally ordered class IDs.

    Stable sort gives deterministic Palm-ID tie-breaking for sorted classes.
    """
    scores = np.asarray(scores)
    classes = np.asarray(classes, dtype=np.int32)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("scores dimension must equal class count")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite class score")
    if not np.array_equal(classes, np.sort(np.unique(classes))):
        raise ValueError("classes must be sorted and unique")
    ranks = np.argsort(scores, axis=1, kind="stable")
    return classes[ranks], np.take_along_axis(scores, ranks, axis=1)


def compute_metrics(
    y_true: np.ndarray, ranked: np.ndarray, classes: np.ndarray | None = None
) -> dict:
    y = np.asarray(y_true, dtype=np.int32)
    ranked = np.asarray(ranked, dtype=np.int32)
    if len(y) != len(ranked) or ranked.ndim != 2:
        raise ValueError("Invalid metrics shapes")
    if classes is None:
        classes = np.sort(np.unique(np.r_[y, ranked.ravel()]))
    if len(y) == 0:
        raise ValueError("Empty test set")
    predicted = ranked[:, 0]
    p, r, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=np.asarray(classes), average="macro", zero_division=0
    )
    rank_position = np.argmax(ranked == y[:, None], axis=1) + 1
    found = np.any(ranked == y[:, None], axis=1)
    rank_position = np.where(found, rank_position, len(classes) + 1)
    result = {
        "n": int(len(y)),
        "classes": int(len(classes)),
        "rank1": float(np.mean(predicted == y)),
        "rank5": float(np.mean(rank_position <= min(5, len(classes)))),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "cmc": {
            str(k): float(np.mean(rank_position <= min(k, len(classes))))
            for k in (1, 2, 5, 10, 20)
            if k <= len(classes)
        },
    }
    return result


def bootstrap_rank1(
    y: np.ndarray, predicted: np.ndarray, *, iterations: int = 1000, seed: int = 42
) -> list[float]:
    """Palm-ID clustered resampling; within-palm probes remain together."""
    y = np.asarray(y)
    hit = (np.asarray(predicted) == y).astype(float)
    ids = np.unique(y)
    per_palm_correct = np.array([hit[y == ident].sum() for ident in ids])
    per_palm_n = np.array([(y == ident).sum() for ident in ids])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ids), size=(iterations, len(ids)))
    ranks = per_palm_correct[samples].sum(axis=1) / per_palm_n[samples].sum(axis=1)
    return [float(v) for v in np.quantile(ranks, [0.025, 0.975])]


def bootstrap_delta(
    y: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> list[float]:
    y = np.asarray(y)
    ids = np.unique(y)
    correct_a = np.asarray(predictions_a) == y
    correct_b = np.asarray(predictions_b) == y
    differences = np.array([(correct_a[y == c].sum() - correct_b[y == c].sum()) for c in ids])
    sizes = np.array([(y == c).sum() for c in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(iterations, len(ids)))
    sampled = differences[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return [float(v) for v in np.quantile(sampled, [0.025, 0.975])]


def matrix(y: np.ndarray, ranked: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, ranked[:, 0], labels=classes)
