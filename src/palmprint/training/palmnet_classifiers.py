"""Fixed exploratory classifier comparison, sharing one Session1-fitted selection."""

import json
import time

import joblib
import numpy as np

from palmprint.features.palmnet import SparseVarianceSelector
from palmprint.models.classical import CollaborativeRidge, LinearSVM, NearestTemplate
from palmprint.training.palmnet import (
    _cache_signature,
    _check_frozen,
    _gallery,
    _paths,
    get_vendor,
    load_csr,
    read_context,
    save_result,
)
from palmprint.utils import atomic_json, hash_file


def train_classifiers(args):
    ws, dest, out, vendor = _paths(args)
    if (dest / "classifiers/freeze.json").exists():
        raise FileExistsError("Classifier experiment is already frozen.")
    if (out / "cache/probe/index.json").exists():
        raise RuntimeError("Freeze the classifier comparison before accessing Session2 ROIs.")
    frame, _, manifest_sha = read_context(ws, args.data_root)
    _, _, info = get_vendor(vendor)
    frozen = _check_frozen(dest, manifest_sha, info)
    if frozen["classifier_plan"] != {
        "selection": "Session1 top variance 1024 columns; L2",
        "linear_svm_C": 1.0,
        "crc_lambda": 0.1,
        "reduced_1nn_control": True,
    }:
        raise RuntimeError("Unexpected classifier plan; do not choose settings using Session2.")
    gallery, labels = _gallery(dest, out, frozen)
    uids = json.loads((dest / "gallery_uids.json").read_text())
    expected = frame[frame.session == "session1"].sort_values(
        ["palm_id", "sample_uid"], kind="stable"
    )
    if uids != expected.sample_uid.tolist() or not np.array_equal(labels, expected.palm_id):
        raise RuntimeError("Gallery UID/label alignment changed.")
    start = time.perf_counter()
    selector = SparseVarianceSelector(n_features=1024).fit(gallery)
    X = selector.transform(gallery)
    selection_seconds = time.perf_counter() - start
    models, times = {}, {}
    for name, model in [
        ("selected_1nn", NearestTemplate()),
        ("linear_svm", LinearSVM(C=1.0)),
        ("crc_rls", CollaborativeRidge(lam=0.1)),
    ]:
        start = time.perf_counter()
        model.fit(X, labels)
        times[name] = time.perf_counter() - start
        models[name] = model
    artifact = dest / "classifiers/models.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"selector": selector, "models": models}, artifact, compress=3)
    atomic_json(
        artifact.parent / "freeze.json",
        {
            "status": "frozen_session1_only",
            "manifest_sha256": manifest_sha,
            "parent_model_sha256": frozen["model_sha256"],
            "plan": frozen["classifier_plan"],
            "selected_columns": selector.columns_.tolist(),
            "source_sha256": hash_file(__file__),
            "selector_source_sha256": hash_file(ws / "src/palmprint/features/palmnet.py"),
            "artifact_sha256": hash_file(artifact),
            "selection_seconds": selection_seconds,
            "fit_seconds": times,
            "note": "Fixed exploratory settings. No CV tuning or Session2 selection. "
            "SVM/CRC must be compared with selected_1nn to separate selection effects.",
        },
    )
    print(
        json.dumps(
            {
                "status": "frozen_session1_only",
                "selected_features": X.shape[1],
                "selection_seconds": selection_seconds,
                "fit_seconds": times,
            },
            indent=2,
        )
    )


def evaluate_classifiers(args):
    ws, dest, out, vendor = _paths(args)
    # Evaluate the primary full-descriptor 1-NN benchmark first.
    if not (out / "nearest_template/metrics.json").exists():
        raise RuntimeError("Complete the primary M5 1-NN benchmark first.")
    frame, _, manifest_sha = read_context(ws, args.data_root)
    _, _, info = get_vendor(vendor)
    frozen = _check_frozen(dest, manifest_sha, info)
    lock = json.loads((dest / "classifiers/freeze.json").read_text())
    artifact = dest / "classifiers/models.joblib"
    if (
        lock["manifest_sha256"] != manifest_sha
        or lock["parent_model_sha256"] != frozen["model_sha256"]
        or hash_file(artifact) != lock["artifact_sha256"]
        or lock["source_sha256"] != hash_file(__file__)
        or lock["selector_source_sha256"] != hash_file(ws / "src/palmprint/features/palmnet.py")
    ):
        raise RuntimeError("Classifier experiment provenance changed.")
    probes = (
        frame[frame.session == "session2"]
        .sort_values(["palm_id", "sample_uid"], kind="stable")
        .reset_index(drop=True)
    )
    cache = out / "cache/probe"
    index = json.loads((cache / "index.json").read_text())
    meta = json.loads((cache / "csr.json").read_text())
    if index["signature"] != _cache_signature(frozen["model_sha256"], probes):
        raise RuntimeError("Probe cache sample/model mismatch.")
    for name, digest in meta["hashes"].items():
        if hash_file(cache / name) != digest:
            raise RuntimeError("Probe CSR cache modified.")
    bundle = joblib.load(artifact)
    start = time.perf_counter()
    X = bundle["selector"].transform(load_csr(cache))
    transform_seconds = time.perf_counter() - start
    for name, model in bundle["models"].items():
        if (out / name / "metrics.json").exists():
            raise FileExistsError(f"Already evaluated: {name}")
        start = time.perf_counter()
        # Limit NearestTemplate's dense pairwise distance matrix as well.
        scores = np.vstack([model.scores(X[i : i + 64]) for i in range(0, len(X), 64)])
        elapsed = time.perf_counter() - start
        save_result(
            out,
            name,
            scores,
            model.classes_,
            probes,
            {
                "manifest_sha256": manifest_sha,
                "model_sha256": frozen["model_sha256"],
                "classifier_artifact_sha256": lock["artifact_sha256"],
                "feature_dim": X.shape[1],
                "selection_fit_session": "session1",
                "hyperparameter_tuning": "none",
                "selection_seconds": lock["selection_seconds"],
                "fit_seconds": lock["fit_seconds"][name],
                "probe_selection_seconds": transform_seconds,
                "matching_seconds": elapsed,
                "score_kind": {
                    "selected_1nn": "min squared Euclidean on selected L2 features",
                    "linear_svm": "negative LinearSVC margin",
                    "crc_rls": "unnormalized CRC_RLS Euclidean residual",
                }[name],
                "interpretation": "Exploratory; includes 1024-column variance selection. "
                "Use selected_1nn as the common-representation control.",
            },
        )
