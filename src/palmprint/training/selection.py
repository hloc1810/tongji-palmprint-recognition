"""Classical cross-validation and model selection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import compute_metrics, ranking
from ..features.extraction import load_features
from ..models.classical import build_model
from ..utils import atomic_json, hash_file, hash_object, read_yaml


def _models_config(ws: Path, override: Path | None = None) -> dict:
    return read_yaml(override or (ws / "configs/models.yaml"))


def _classical_source_hash(ws: Path) -> str:
    """Fingerprint the code that defines classical model fitting and evaluation."""
    source_dir = Path(__file__).resolve().parents[1]
    paths = (
        "data/dataset.py",
        "data/preparation.py",
        "features/compcode.py",
        "features/extraction.py",
        "models/classical.py",
        "training/selection.py",
        "training/classical.py",
        "evaluation/metrics.py",
        "reporting/evidence.py",
        "utils.py",
    )
    return hash_object({name: hash_file(source_dir / name) for name in paths})


def _selection_paths(ws: Path, model_name: str, rep: str = "compcode") -> tuple[Path, Path]:
    folder = ws / "outputs/cv"
    return folder / f"{rep}_{model_name}.csv", folder / f"{rep}_{model_name}_selection.json"


def cross_validate(
    ws: Path,
    model_name: str,
    cfg_path: Path,
    *,
    rep: str = "compcode",
    model_cfg: Path | None = None,
) -> dict:
    """5-fold class-balanced CV, with held-out per-image OOF evidence.

    Session2 ROI pixels are never loaded or extracted by this function.
    """
    if (ws / "artifacts/protocol_lock.json").exists():
        raise RuntimeError("Protocol locked: CV is immutable")
    if model_name not in ("knn", "svm", "crc"):
        raise ValueError("Model must be knn, svm or crc")
    config = _models_config(ws, model_cfg)
    params = [float(v) for v in config[model_name]["parameters"]]
    if not params or len(params) != len(set(params)):
        raise ValueError("Parameter grid empty or duplicated")
    df, x1, signature = load_features(ws, cfg_path, rep, sessions=("session1",))
    labels = df.palm_id.to_numpy(dtype=np.int32)
    folds = sorted(df.fold.unique())
    scores, predictions = [], []
    for p in params:
        for fold in folds:
            is_val = df.fold.to_numpy() == fold
            started_fit = time.perf_counter()
            model = build_model(model_name, p).fit(x1[~is_val], labels[~is_val])
            fold_fit_s = time.perf_counter() - started_fit
            started_infer = time.perf_counter()
            ordered, ordered_scores = ranking(model.scores(x1[is_val]), model.classes_)
            fold_predict_s = time.perf_counter() - started_infer
            y_true = labels[is_val]
            met = compute_metrics(y_true, ordered, model.classes_)
            scores.append(
                {
                    "model": model_name,
                    "representation": rep,
                    "parameter": p,
                    "fold": int(fold),
                    "train_n": int((~is_val).sum()),
                    "val_n": int(is_val.sum()),
                    "fit_seconds": fold_fit_s,
                    "val_predict_seconds": fold_predict_s,
                    "rank1": met["rank1"],
                    "rank5": met["rank5"],
                    "macro_f1": met["macro_f1"],
                    "feature_signature": signature,
                }
            )
            for j, sample in enumerate(df.loc[is_val].itertuples(index=False)):
                rank_true = int(np.flatnonzero(ordered[j] == y_true[j])[0]) + 1
                predictions.append(
                    {
                        "sample_uid": sample.sample_uid,
                        "true_palm_id": int(y_true[j]),
                        "parameter": p,
                        "fold": int(fold),
                        "pred_palm_id": int(ordered[j, 0]),
                        "correct": int(ordered[j, 0] == y_true[j]),
                        "true_rank": rank_true,
                        "top5_palm_ids": json.dumps(ordered[j, :5].tolist()),
                        "top5_scores": json.dumps(ordered_scores[j, :5].tolist()),
                    }
                )
    table = pd.DataFrame(scores)
    out, sel_path = _selection_paths(ws, model_name, rep)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary = table.groupby("parameter")["rank1"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values(["mean", "parameter"], ascending=[False, True])
    best = float(summary.iloc[0].parameter)
    selected_oof = pd.DataFrame(predictions)
    selected_oof = selected_oof.loc[selected_oof.parameter == best].sort_values("sample_uid")
    if not selected_oof.sample_uid.is_unique or len(selected_oof) != len(df):
        raise RuntimeError("Each session1 image must have exactly one selected OOF prediction")
    oof_path = out.parent / f"{rep}_{model_name}_oof_predictions.csv"
    selected_oof.to_csv(oof_path, index=False)
    folds_path = ws / "data/splits/fold_assignments.csv"
    choice = {
        "model": model_name,
        "representation": rep,
        "best_parameter": best,
        "parameter_grid": params,
        "mean_cv_rank1": float(summary.iloc[0]["mean"]),
        "std_cv_rank1": float(summary.iloc[0]["std"]),
        "n_folds": len(folds),
        "feature_signature": signature,
        "cv_csv_sha256": hash_file(out),
        "oof_csv_sha256": hash_file(oof_path),
        "fold_assignments_sha256": hash_file(folds_path),
        "model_config_sha256": hash_file(model_cfg or ws / "configs/models.yaml"),
        "source_sha256": _classical_source_hash(ws),
        "selection": "maximize mean session1 CV Rank-1; tie: smallest parameter",
    }
    atomic_json(sel_path, choice)
    return choice
