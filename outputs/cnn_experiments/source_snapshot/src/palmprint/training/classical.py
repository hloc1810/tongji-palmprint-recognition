"""Final classical training, protocol lock, test and inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from ..data.dataset import read_gray
from ..data.preparation import load_manifest
from ..evaluation.metrics import bootstrap_rank1, compute_metrics, matrix, ranking
from ..features.compcode import CompCodeConfig, CompCodeExtractor
from ..features.extraction import feature_config, load_features
from ..models.classical import build_model, model_definition
from ..reporting.evidence import write_prediction_outputs
from ..utils import atomic_json, env_info, hash_file
from .selection import _classical_source_hash, _selection_paths


def final_train(ws: Path, model_name: str, cfg_path: Path) -> dict:
    if (ws / "artifacts/protocol_lock.json").exists():
        raise RuntimeError("Protocol locked: cannot refit or replace model")
    if (ws / "artifacts" / f"{model_name}_freeze.json").exists():
        raise RuntimeError("Frozen run exists. Use a separate workspace for new experiments")
    feature_started = time.perf_counter()
    df, X, signature = load_features(ws, cfg_path, sessions=("session1",))
    feature_loading_seconds = time.perf_counter() - feature_started
    cv_path, sel_path = _selection_paths(ws, model_name)
    selected = json.loads(sel_path.read_text(encoding="utf-8"))
    oof_path = ws / "outputs/cv" / f"compcode_{model_name}_oof_predictions.csv"
    if selected["feature_signature"] != signature:
        raise RuntimeError("CV feature config differs from final config")
    if hash_file(cv_path) != selected["cv_csv_sha256"] or hash_file(oof_path) != selected["oof_csv_sha256"]:
        raise RuntimeError("CV predictions/results were modified after selection")
    if hash_file(ws / "data/splits/fold_assignments.csv") != selected["fold_assignments_sha256"]:
        raise RuntimeError("CV fold assignments changed")
    if hash_file(ws / "configs/models.yaml") != selected["model_config_sha256"]:
        raise RuntimeError("Model hyperparameter grid changed")
    if _classical_source_hash(ws) != selected["source_sha256"]:
        raise RuntimeError("Classical code changed after CV; rerun CV")
    started_fit = time.perf_counter()
    model = build_model(model_name, selected["best_parameter"]).fit(
        X, df.palm_id.to_numpy(dtype=np.int32))
    training_fit_seconds = time.perf_counter() - started_fit
    ws_path = ws / "artifacts" / f"{model_name}.joblib"
    ws_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_config": feature_config(cfg_path).__dict__,
                 "feature_signature": signature, "model_name": model_name,
                 "model_definition": model_definition(model)}, ws_path, compress=3)
    info = {"model": model_name, "selected": selected,
            "model_definition": model_definition(model),
            "selected_sha256": hash_file(sel_path),
            "artifact_sha256": hash_file(ws_path),
            "manifest_sha256": hash_file(ws / "data/metadata/manifest.csv"),
            "feature_signature": signature,
            "source_sha256": _classical_source_hash(ws),
            "classes": model.classes_.tolist(), "train_n": int(len(df)),
            "feature_loading_seconds": feature_loading_seconds,
            "training_fit_seconds": training_fit_seconds,
            "test_n": int(len(load_manifest(ws)[0]) - len(df)),
            "test_status": "not_evaluated", "environment": env_info()}
    atomic_json(ws / "artifacts" / f"{model_name}_freeze.json", info)
    return {"model": model_name, "n_classes": len(model.classes_),
            "artifact": str(ws_path), "training_fit_seconds": training_fit_seconds,
            "test_status": "not_evaluated"}


def lock_protocol(ws: Path, *, models: tuple[str, ...] = ("knn", "svm", "crc")) -> dict:
    """Seal all classical candidate fits before permitting any session2 test inference."""
    dest = ws / "artifacts/protocol_lock.json"
    if dest.exists():
        raise RuntimeError("Protocol already locked")
    if tuple(models) != ("knn", "svm", "crc"):
        raise ValueError("All three planned models must be frozen together")
    if any((ws / "outputs/runs" / name / "metrics.json").exists() for name in models):
        raise RuntimeError("Cannot lock after test results already exist")
    data, _ = load_manifest(ws)
    sha = _classical_source_hash(ws)
    models_info = {}
    for name in models:
        path = ws / "artifacts" / f"{name}_freeze.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        artifact_path = ws / "artifacts" / f"{name}.joblib"
        if (hash_file(artifact_path) != frozen["artifact_sha256"] or
            frozen["source_sha256"] != sha or
            frozen["manifest_sha256"] != hash_file(ws / "data/metadata/manifest.csv")):
            raise RuntimeError(f"Frozen {name} model/code/data mismatch")
        models_info[name] = {"freeze_sha256": hash_file(path),
                             "artifact_sha256": hash_file(artifact_path),
                             "best_parameter": frozen["selected"]["best_parameter"]}
    lock = {"status": "locked_before_test", "model_set": list(models),
            "manifest_sha256": hash_file(ws / "data/metadata/manifest.csv"),
            "source_sha256": sha,
            "session1_images": int((data.session == "session1").sum()),
            "session2_images": int((data.session == "session2").sum()),
            "models": models_info}
    atomic_json(dest, lock)
    return lock


def evaluate_final(ws: Path, model_name: str, cfg_path: Path) -> dict:
    """Evaluate a frozen experiment, after sealing all classical candidates."""
    lock_file = ws / "artifacts/protocol_lock.json"
    if not lock_file.exists():
        raise RuntimeError("All classical models must be frozen first: run 'lock'")
    locked = json.loads(lock_file.read_text(encoding="utf-8"))
    if locked["source_sha256"] != _classical_source_hash(ws):
        raise RuntimeError("Source code changed after protocol lock")
    if model_name not in locked["models"]:
        raise RuntimeError("Model not included in protocol lock")
    frozen_path = ws / "artifacts" / f"{model_name}_freeze.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if hash_file(frozen_path) != locked["models"][model_name]["freeze_sha256"]:
        raise RuntimeError("Frozen model metadata changed after lock")
    outdir = ws / "outputs/runs" / model_name
    if (outdir / "metrics.json").exists():
        raise RuntimeError("Test already evaluated. Results are immutable; no re-tuning on test")
    model_path = ws / "artifacts" / f"{model_name}.joblib"
    if hash_file(model_path) != frozen["artifact_sha256"]:
        raise RuntimeError("Model artifact modified after freeze")
    if hash_file(ws / "data/metadata/manifest.csv") != frozen["manifest_sha256"]:
        raise RuntimeError("Dataset changed after freeze")
    test_feature_started = time.perf_counter()
    df, X, _ = load_features(ws, cfg_path, sessions=("session2",))
    test_feature_loading_seconds = time.perf_counter() - test_feature_started
    item = joblib.load(model_path)  # ONLY load locally-generated trusted artifacts.
    model = item["model"]
    # CompCode configuration is identical even though cache signatures differ by session.
    if feature_config(cfg_path).__dict__ != item["feature_config"]:
        raise RuntimeError("Feature configuration changed after final training")
    start = time.perf_counter()
    scores = np.concatenate([model.scores(z) for z in np.array_split(
        X, max(1, (len(X) + 31) // 32)) if len(z)], axis=0)
    classifier_elapsed = time.perf_counter() - start
    ordered, ordered_scores = ranking(scores, model.classes_)
    true_ids = df.palm_id.to_numpy(dtype=np.int32)
    metric = compute_metrics(true_ids, ordered, model.classes_)
    metric["rank1_ci95_palm_bootstrap"] = bootstrap_rank1(true_ids, ordered[:, 0])
    metric["model"] = model_name
    metric["evaluation"] = "session1-trained vs session2-test, closed-set"
    metric["matching_elapsed_total_s"] = classifier_elapsed
    metric["feature_loading_elapsed_s"] = test_feature_loading_seconds
    metric["training_fit_seconds"] = frozen["training_fit_seconds"]
    metric["matching_ms_per_probe_batch_amortized"] = 1000 * classifier_elapsed / len(df)
    metric["model_definition"] = model_definition(model)
    metric["score_kind"] = metric["model_definition"]["score_kind"]
    metric["manifest_sha256"] = frozen["manifest_sha256"]
    metric["artifact_sha256"] = frozen["artifact_sha256"]
    metric["feature_definition"] = item["feature_config"]
    metric["code_sha256"] = _classical_source_hash(ws)
    metric["warning"] = "Historical HOG pilot examined some session2 IDs; test not fully untouched."
    outdir.mkdir(parents=True, exist_ok=True)
    write_prediction_outputs(outdir, df, ordered, ordered_scores, metric, model.classes_,
                             unsorted_scores=scores)
    np.save(outdir / "confusion_matrix.npy", matrix(true_ids, ordered, model.classes_))
    atomic_json(outdir / "metrics.json", metric)
    return metric


def predict_one(ws: Path, model_name: str, image_path: Path) -> dict:
    """Prediction uses image pixels only; a filename never supplies a label."""
    artifact_path = ws / "artifacts" / f"{model_name}.joblib"
    freeze_path = ws / "artifacts" / f"{model_name}_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if hash_file(artifact_path) != frozen["artifact_sha256"]:
        raise RuntimeError("Demo artifact hash differs from frozen model")
    bundle = joblib.load(artifact_path)  # Locally produced trusted file only.
    extractor = CompCodeExtractor(CompCodeConfig(**bundle["feature_config"]))
    start = time.perf_counter()
    feature = extractor.transform(read_gray(image_path, extractor.config.image_size))
    model = bundle["model"]
    scores = model.scores(feature[None, :])
    ids, values = ranking(scores, model.classes_)
    return {"model": model_name, "predicted_palm_id": int(ids[0, 0]),
            "top5": ids[0, :5].tolist(), "scores_lower_is_better": values[0, :5].tolist(),
            "elapsed_ms": 1000*(time.perf_counter()-start),
            "score_kind": {"knn": "squared_euclidean", "svm": "negative_margin",
                           "crc": "reconstruction_residual"}[model_name]}
