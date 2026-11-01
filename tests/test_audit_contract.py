"""Adversarial regression tests: CV isolation, immutable protocol and report provenance."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from palmprint.data.dataset import generate_synthetic_roi
from palmprint.data.preparation import prepare
from palmprint.features.extraction import load_features
from palmprint.reporting.evidence import verify_run_evidence
from palmprint.reporting.summary import generate_report
from palmprint.training.classical import evaluate_final, final_train, lock_protocol
from palmprint.training.selection import cross_validate


def make_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", ws / "configs")
    data = tmp_path / "fake"
    generate_synthetic_roi(data, palms=3, samples=4)
    prepare(ws, data, strict=False, folds=2)
    return ws, data


def test_cv_never_reads_or_caches_session2(tmp_path, monkeypatch):
    ws, _ = make_workspace(tmp_path)
    import palmprint.features.extraction as extraction

    real_reader = extraction.read_gray

    def forbid_test(path, *args, **kwargs):
        if "session2" in str(path):
            raise AssertionError("CV read test ROI pixels")
        return real_reader(path, *args, **kwargs)

    monkeypatch.setattr(extraction, "read_gray", forbid_test)
    cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    final_train(ws, "knn", ws / "configs/compcode.yaml")
    assert not list((ws / "outputs/cache").glob("*session2*"))
    with pytest.raises(RuntimeError, match="protocol lock"):
        load_features(ws, ws / "configs/compcode.yaml", sessions=("session2",))
    oof = pd.read_csv(ws / "outputs/cv/compcode_knn_oof_predictions.csv")
    assert oof.sample_uid.is_unique and len(oof) == 12
    assert set(oof.fold) == {0, 1}


def test_frozen_protocol_and_scores_are_self_consistent(tmp_path):
    ws, _ = make_workspace(tmp_path)
    cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    final_train(ws, "knn", ws / "configs/compcode.yaml")
    with pytest.raises(RuntimeError, match="run 'lock'"):
        evaluate_final(ws, "knn", ws / "configs/compcode.yaml")
    for name in ("svm", "crc"):
        cross_validate(ws, name, ws / "configs/compcode.yaml")
        final_train(ws, name, ws / "configs/compcode.yaml")
    lock_protocol(ws)
    with pytest.raises(RuntimeError, match="locked"):
        cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    with pytest.raises(RuntimeError, match="locked"):
        final_train(ws, "knn", ws / "configs/compcode.yaml")
    with pytest.raises(RuntimeError, match="locked"):
        prepare(ws, ws.parent / "fake", strict=False, folds=2)
    for name in ("knn", "svm", "crc"):
        evaluate_final(ws, name, ws / "configs/compcode.yaml")
        assert verify_run_evidence(ws / "outputs/runs" / name)["verified"]
        run = ws / "outputs/runs" / name
        scores = np.load(run / "all_class_scores.npz", allow_pickle=False)
        assert scores["scores"].shape == (12, 3)
        pred = pd.read_csv(run / "predictions.csv")
        assert {"true_rank", "top5_scores", "top1_score", "score_gap_top2_minus_top1"} <= set(
            pred.columns
        )
        assert len(pd.read_csv(run / "per_class_metrics.csv")) == 3
        assert len(pd.read_csv(run / "cmc_full.csv")) == 3
        assert (run / "confusion_matrix.npy").exists()
    generate_report(ws)
    assert (ws / "reports/tables/paired_rank1_differences.csv").exists()
    assert (ws / "reports/tables/evidence_verification.csv").exists()
    assert (ws / "reports/figures/rank1_confidence_intervals.png").exists()
    bad = ws / "outputs/runs/knn/predictions.csv"
    tab = pd.read_csv(bad)
    tab.loc[0, "pred_palm_id"] = 3 if tab.loc[0, "pred_palm_id"] != 3 else 2
    tab.to_csv(bad, index=False)
    with pytest.raises(RuntimeError, match="differ from class scores"):
        generate_report(ws)


def test_cv_grid_change_is_rejected(tmp_path):
    ws, _ = make_workspace(tmp_path)
    cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    cfg = ws / "configs/models.yaml"
    cfg.write_text(cfg.read_text() + "\n# silent parameter configuration change\n")
    with pytest.raises(RuntimeError, match="grid changed"):
        final_train(ws, "knn", ws / "configs/compcode.yaml")


def test_raw_data_mutation_is_detected_even_if_cached(tmp_path):
    ws, root = make_workspace(tmp_path)
    cross_validate(ws, "knn", ws / "configs/compcode.yaml")
    file = root / "session1/00001.png"
    frame = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    frame[0, 0] = np.uint8((int(frame[0, 0]) + 1) % 256)
    assert cv2.imwrite(str(file), frame)
    with pytest.raises(RuntimeError, match="changed after dataset audit"):
        final_train(ws, "knn", ws / "configs/compcode.yaml")


def test_original_image_must_not_be_resized_into_roi(tmp_path):
    from palmprint.data.dataset import read_gray

    wrong = tmp_path / "original.png"
    assert cv2.imwrite(str(wrong), np.zeros((600, 800), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected Official ROI"):
        read_gray(wrong)
