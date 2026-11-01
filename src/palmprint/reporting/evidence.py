"""Saved score evidence and consistency verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import atomic_json


def write_prediction_outputs(
    outdir: Path,
    probes: pd.DataFrame,
    ranked: np.ndarray,
    ranked_scores: np.ndarray,
    metrics: dict,
    classes: np.ndarray,
    *,
    unsorted_scores: np.ndarray,
) -> None:
    """Export complete per-probe evidence; preserve class score order for verification."""
    from sklearn.metrics import precision_recall_fscore_support

    classes = np.asarray(classes, dtype=np.int32)
    if unsorted_scores.shape != (len(probes), len(classes)):
        raise ValueError("Expected one score for each sample and every Palm ID")
    true_ids = probes.palm_id.to_numpy(dtype=np.int32)
    if not np.isin(true_ids, classes).all():
        raise ValueError("A test label is missing from the class-order matrix")
    true_pos = np.searchsorted(classes, true_ids)
    true_scores = unsorted_scores[np.arange(len(true_ids)), true_pos]
    rank_positions = np.argmax(ranked == true_ids[:, None], axis=1) + 1
    np.savez_compressed(
        outdir / "all_class_scores.npz",
        scores=unsorted_scores.astype(np.float64),
        classes=classes,
        sample_uid=np.asarray(probes.sample_uid.tolist(), dtype=str),
        true_palm_id=true_ids,
        description=np.asarray("lower scores rank better; class-order columns"),
    )
    rows = []
    for k, (_, sample) in enumerate(probes.iterrows()):
        rows.append(
            {
                "sample_uid": str(sample.sample_uid),
                "true_palm_id": int(sample.palm_id),
                "pred_palm_id": int(ranked[k, 0]),
                "correct": int(ranked[k, 0] == sample.palm_id),
                "true_rank": int(rank_positions[k]),
                "rank5_correct": int(rank_positions[k] <= min(5, len(classes))),
                "top1_score": float(ranked_scores[k, 0]),
                "top2_score": float(ranked_scores[k, 1]) if len(classes) > 1 else None,
                "score_gap_top2_minus_top1": float(ranked_scores[k, 1] - ranked_scores[k, 0])
                if len(classes) > 1
                else None,
                "true_class_score": float(true_scores[k]),
                "top5_palm_ids": json.dumps(ranked[k, :5].tolist()),
                "top5_scores": json.dumps(ranked_scores[k, :5].tolist()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "predictions.csv", index=False)
    table.loc[table.correct == 0].to_csv(outdir / "errors.csv", index=False)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, ranked[:, 0], labels=classes, average=None, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "palm_id": classes,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": np.array(
                [((true_ids == c) & (ranked[:, 0] == c)).sum() for c in classes], dtype=int
            ),
        }
    )
    per_class["errors"] = per_class.support - per_class.correct
    per_class.to_csv(outdir / "per_class_metrics.csv", index=False)
    cmc = np.array([(rank_positions <= k).mean() for k in range(1, len(classes) + 1)])
    pd.DataFrame({"rank_k": np.arange(1, len(classes) + 1), "identification_rate": cmc}).to_csv(
        outdir / "cmc_full.csv", index=False
    )
    atomic_json(outdir / "class_order.json", classes.tolist())


def verify_run_evidence(
    run_dir: Path,
    *,
    expected_probes: pd.DataFrame | None = None,
    manifest_sha256: str | None = None,
) -> dict:
    """Recompute all principal metrics from saved score matrix, not from cached metrics."""
    from sklearn.metrics import precision_recall_fscore_support

    from ..evaluation.metrics import bootstrap_rank1, compute_metrics, matrix, ranking

    metric = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "all_class_scores.npz", allow_pickle=False) as data:
        score = data["scores"].astype(np.float64)
        classes = data["classes"].astype(np.int32)
        sample_uid = data["sample_uid"].tolist()
        y = data["true_palm_id"].astype(np.int32)
    pred = pd.read_csv(run_dir / "predictions.csv")
    if (
        len(pred) != len(y)
        or len(set(sample_uid)) != len(sample_uid)
        or not np.isin(y, classes).all()
        or score.shape != (len(y), len(classes))
        or pred.sample_uid.tolist() != sample_uid
        or not np.array_equal(pred.true_palm_id.to_numpy(), y)
    ):
        raise RuntimeError(f"Inconsistent score evidence for {run_dir.name}")
    if expected_probes is not None:
        if (
            expected_probes.sample_uid.duplicated().any()
            or expected_probes.sample_uid.tolist() != sample_uid
            or not np.array_equal(expected_probes.palm_id.to_numpy(), y)
            or not np.array_equal(np.sort(expected_probes.palm_id.unique()), classes)
        ):
            raise RuntimeError(f"Score evidence differs from session2 manifest: {run_dir.name}")
    if manifest_sha256 is not None and metric.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError(f"Run manifest hash mismatch: {run_dir.name}")
    stored_classes = json.loads((run_dir / "class_order.json").read_text(encoding="utf-8"))
    if stored_classes != classes.tolist():
        raise RuntimeError(f"Class-order evidence mismatch: {run_dir.name}")
    rank, ranked_scores = ranking(score, classes)
    rank_pos = np.argmax(rank == y[:, None], axis=1) + 1
    if not np.array_equal(pred.pred_palm_id.to_numpy(), rank[:, 0]) or not np.array_equal(
        pred.true_rank.to_numpy(), rank_pos
    ):
        raise RuntimeError(f"Reported predictions/ranks differ from class scores: {run_dir.name}")
    recomputed = compute_metrics(y, rank, classes)
    for key in ("n", "classes", "rank1", "rank5", "macro_precision", "macro_recall", "macro_f1"):
        if not np.isclose(float(metric[key]), float(recomputed[key]), atol=1e-7, rtol=0):
            raise RuntimeError(f"Reported {key} differs from raw class scores: {run_dir.name}")
    ci = np.asarray(metric["rank1_ci95_palm_bootstrap"], dtype=float)
    if ci.shape != (2,) or not np.allclose(ci, bootstrap_rank1(y, rank[:, 0]), atol=1e-8, rtol=0):
        raise RuntimeError(f"Reported bootstrap CI differs from class scores: {run_dir.name}")
    if metric.get("cmc") != recomputed["cmc"]:
        raise RuntimeError(f"Reported CMC differs from class scores: {run_dir.name}")
    for field, expected in (
        ("correct", (rank[:, 0] == y).astype(int)),
        ("rank5_correct", (rank_pos <= min(5, len(classes))).astype(int)),
        ("top1_score", ranked_scores[:, 0]),
        ("true_class_score", score[np.arange(len(y)), np.searchsorted(classes, y)]),
    ):
        if not np.allclose(pred[field].to_numpy(), expected, atol=1e-8, rtol=1e-10):
            raise RuntimeError(f"Reported {field} differs from class scores: {run_dir.name}")
    if len(classes) > 1:
        for field, expected in (
            ("top2_score", ranked_scores[:, 1]),
            ("score_gap_top2_minus_top1", ranked_scores[:, 1] - ranked_scores[:, 0]),
        ):
            if not np.allclose(pred[field].to_numpy(), expected, atol=1e-8, rtol=1e-10):
                raise RuntimeError(f"Reported {field} differs from class scores: {run_dir.name}")
    if not np.array_equal(
        np.array([json.loads(v) for v in pred.top5_palm_ids]), rank[:, :5]
    ) or not np.allclose(
        np.array([json.loads(v) for v in pred.top5_scores]),
        ranked_scores[:, :5],
        atol=1e-8,
        rtol=1e-10,
    ):
        raise RuntimeError(f"Reported Top-5 differs from class scores: {run_dir.name}")
    per_class = pd.read_csv(run_dir / "per_class_metrics.csv")
    if len(per_class) != len(classes) or not np.array_equal(per_class.palm_id.to_numpy(), classes):
        raise RuntimeError(f"Wrong per-class metrics class order: {run_dir.name}")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, rank[:, 0], labels=classes, average=None, zero_division=0
    )
    for field, expected in (
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
        ("support", support),
        ("correct", np.diag(matrix(y, rank, classes))),
        ("errors", support - np.diag(matrix(y, rank, classes))),
    ):
        if not np.allclose(per_class[field].to_numpy(), expected, atol=1e-8):
            raise RuntimeError(f"Incorrect per-class {field}: {run_dir.name}")
    real_cm = matrix(y, rank, classes)
    stored_cm = np.load(run_dir / "confusion_matrix.npy", allow_pickle=False)
    if not np.array_equal(real_cm, stored_cm):
        raise RuntimeError(f"Incorrect confusion matrix: {run_dir.name}")
    errors = pd.read_csv(run_dir / "errors.csv")
    true_error = pred.loc[rank[:, 0] != y]
    if (
        len(errors) != len(true_error)
        or errors.sample_uid.tolist() != true_error.sample_uid.tolist()
        or not np.array_equal(
            errors.pred_palm_id.to_numpy(dtype=int), true_error.pred_palm_id.to_numpy(dtype=int)
        )
    ):
        raise RuntimeError(f"Error-list mismatch: {run_dir.name}")
    try:
        pd.testing.assert_frame_equal(
            errors.reset_index(drop=True),
            true_error.reset_index(drop=True),
            check_dtype=False,
            atol=1e-8,
            rtol=1e-10,
        )
    except AssertionError as exc:
        raise RuntimeError(f"Error-list content mismatch: {run_dir.name}") from exc
    curve = pd.read_csv(run_dir / "cmc_full.csv")
    full_rates = np.array([(rank_pos <= k).mean() for k in range(1, len(classes) + 1)])
    if (
        len(curve) != len(classes)
        or not np.array_equal(curve.rank_k.to_numpy(), np.arange(1, len(classes) + 1))
        or not np.allclose(curve.identification_rate.to_numpy(), full_rates, atol=1e-8)
    ):
        raise RuntimeError(f"Corrupted full CMC curve: {run_dir.name}")
    return {
        "model": run_dir.name,
        "samples": len(y),
        "classes": len(classes),
        "rank1": recomputed["rank1"],
        "rank5": recomputed["rank5"],
        "verified": True,
    }
