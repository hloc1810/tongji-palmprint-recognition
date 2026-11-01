"""Scientific evidence regressions: manifest alignment, cache and forged report fields."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from palmprint.data.dataset import generate_synthetic_roi
from palmprint.data.preparation import load_manifest, prepare
from palmprint.features.extraction import load_features
from palmprint.reporting.evidence import verify_run_evidence
from palmprint.reporting.summary import generate_report
from palmprint.training.classical import evaluate_final, final_train, lock_protocol
from palmprint.training.selection import cross_validate
from palmprint.utils import hash_file


@pytest.fixture(scope="module")
def completed_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence_fixture")
    ws = root / "workspace"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", ws / "configs")
    raw = root / "synthetic"
    generate_synthetic_roi(raw, palms=3, samples=4)
    prepare(ws, raw, strict=False, folds=2)
    # A valid, deliberately shuffled manifest rules out np.repeat labels and
    # assumptions that image rows are grouped by Palm ID.
    path = ws / "data/metadata/manifest.csv"
    manifest = pd.read_csv(path, dtype={"stem": "string"}).sample(frac=1, random_state=7)
    manifest.to_csv(path, index=False)
    info_path = ws / "data/metadata/local_dataset.json"
    info = json.loads(info_path.read_text())
    info["manifest_sha256"] = hash_file(path)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    for name in ("knn", "svm", "crc"):
        cross_validate(ws, name, ws / "configs/compcode.yaml")
        final_train(ws, name, ws / "configs/compcode.yaml")
    lock_protocol(ws)
    for name in ("knn", "svm", "crc"):
        evaluate_final(ws, name, ws / "configs/compcode.yaml")
    return ws


@pytest.fixture
def workspace(completed_workspace, tmp_path):
    ws = tmp_path / "workspace"
    shutil.copytree(completed_workspace, ws)
    return ws


def test_report_uses_manifest_order_and_labels_and_marks_fixture(workspace):
    manifest, _ = load_manifest(workspace)
    probes = manifest[manifest.session == "session2"]
    result = generate_report(workspace)
    assert set(result.data_scope) == {"NON_STRICT_fixture_or_subset"}
    assert result["method"].str.len().gt(0).all()
    assert set(result.model) == {"knn", "svm", "crc"}
    for model in result.model:
        pred = pd.read_csv(workspace / f"outputs/runs/{model}/predictions.csv")
        assert pred.sample_uid.tolist() == probes.sample_uid.tolist()
        assert pred.true_palm_id.tolist() == probes.palm_id.tolist()


@pytest.mark.parametrize(
    "field",
    [
        "rank1",
        "n",
        "classes",
        "ci",
        "correct",
        "top5",
        "class_order",
        "manifest_hash",
        "model_definition",
        "artifact",
        "score_kind",
    ],
)
def test_report_rejects_corrupted_evidence(workspace, field):
    run = workspace / "outputs/runs/knn"
    if field in ("rank1", "n", "classes", "ci", "manifest_hash", "model_definition", "score_kind"):
        path = run / "metrics.json"
        metric = json.loads(path.read_text())
        if field == "ci":
            metric["rank1_ci95_palm_bootstrap"] = [-1, 2]
        elif field == "manifest_hash":
            metric["manifest_sha256"] = "wrong"
        elif field == "model_definition":
            metric["model_definition"]["method"] = "unmeasured baseline"
        elif field == "score_kind":
            metric["score_kind"] = "probability"
        else:
            metric[field] += 0.123
        path.write_text(json.dumps(metric), encoding="utf-8")
    elif field == "class_order":
        (run / "class_order.json").write_text("[3, 2, 1]")
    elif field == "artifact":
        artifact = workspace / "artifacts/knn.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    else:
        path = run / "predictions.csv"
        pred = pd.read_csv(path)
        if field == "correct":
            pred.loc[0, "correct"] = 1 - pred.loc[0, "correct"]
        else:
            pred.loc[0, "top5_palm_ids"] = "[600, 2, 3]"
        pred.to_csv(path, index=False)
    with pytest.raises(RuntimeError):
        generate_report(workspace)
    assert not (workspace / "reports/tables/model_comparison.csv").exists()


def test_even_internally_consistent_labels_must_match_manifest(workspace):
    run = workspace / "outputs/runs/knn"
    manifest, info = load_manifest(workspace)
    probes = manifest[manifest.session == "session2"].copy()
    probes["palm_id"] = probes.palm_id.map({1: 2, 2: 3, 3: 1})
    # Scores and exported predictions are unchanged and agree with one another;
    # supplying a different audited ground truth must still fail.
    with pytest.raises(RuntimeError, match="session2 manifest"):
        verify_run_evidence(run, expected_probes=probes, manifest_sha256=info["manifest_sha256"])


@pytest.mark.parametrize("corruption", ["rows", "columns", "uid"])
def test_cache_shape_and_order_are_checked(workspace, corruption):
    cfg = workspace / "configs/compcode.yaml"
    _, x, signature = load_features(workspace, cfg)
    path = workspace / f"outputs/cache/compcode_session1_{signature[:16]}.npz"
    with np.load(path, allow_pickle=False) as data:
        uid = data["uid"].copy()
    if corruption == "rows":
        x = x[:-1]
    elif corruption == "columns":
        x = x[:, :-1]
    else:
        uid = uid[::-1]
    np.savez_compressed(path, X=x, uid=uid, signature=signature)
    with pytest.raises(RuntimeError, match="Invalid cache"):
        load_features(workspace, cfg)
