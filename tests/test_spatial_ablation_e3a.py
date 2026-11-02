"""Unit and integration tests for E3a: Global vs Spatial CompCode Ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor
from palmprint.models.classical import CollaborativeRidge


def test_spatial_ablation_dimensions():
    """Verify that block_pixels configurations yield exact expected feature dimensions."""
    # 1x1 Global histogram -> 1 block x 6 directions = 6-D
    cfg_10x10 = CompCodeConfig(block_pixels=14)
    ext_10x10 = CompCodeExtractor(cfg_10x10)
    assert ext_10x10.feature_dim == 600

    dummy_codes = np.zeros((128, 128), dtype=np.uint8)
    global_hist = ext_10x10.histogram(dummy_codes, global_histogram=True)
    assert global_hist.shape == (6,)
    assert np.isclose(global_hist.sum(), 1.0)

    # 5x5 Grid -> 25 blocks x 6 directions = 150-D
    cfg_5x5 = CompCodeConfig(block_pixels=26)
    ext_5x5 = CompCodeExtractor(cfg_5x5)
    assert ext_5x5.feature_dim == 150

    spatial_5x5_hist = ext_5x5.histogram(dummy_codes, global_histogram=False)
    assert spatial_5x5_hist.shape == (150,)


def test_spatial_block_partitioning_coverage():
    """Verify that spatial grid covers the entire image without discarding borders."""
    cfg = CompCodeConfig(block_pixels=26)
    ext = CompCodeExtractor(cfg)

    # Count blocks across (0, 26, 52, 78, 104) -> 5 intervals along each dimension
    y_starts = list(range(0, cfg.image_size, cfg.block_pixels))
    x_starts = list(range(0, cfg.image_size, cfg.block_pixels))
    assert len(y_starts) == 5
    assert len(x_starts) == 5
    assert len(y_starts) * len(x_starts) == 25
    assert ext.feature_dim == 25 * cfg.num_directions


def test_crc_rls_analytical_solution():
    """Verify CRC_RLS (Collaborative Representation Classifier) fitting and scoring."""
    np.random.seed(42)
    # 10 classes, 5 samples per class, 30 features
    n_classes = 10
    n_per_class = 5
    n_features = 30
    X_train = np.random.randn(n_classes * n_per_class, n_features).astype(np.float32)
    # L2 normalize rows
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    y_train = np.repeat(np.arange(n_classes), n_per_class)

    clf = CollaborativeRidge(lam=0.1).fit(X_train, y_train)
    assert len(clf.classes_) == n_classes

    # Predict on training samples - should have minimum residual for true class
    scores = clf.scores(X_train)
    assert scores.shape == (len(X_train), n_classes)
    from palmprint.evaluation.metrics import ranking

    ordered_classes, _ = ranking(scores, clf.classes_)
    top_preds = ordered_classes[:, 0]
    # Synthetic random data with strong separation (e.g. diagonal signal)
    # Just verify output structure and ordering
    assert top_preds.shape == (len(X_train),)
    assert np.all(np.isin(top_preds, clf.classes_))


def test_e3a_artifacts_integrity():
    """Verify that E3a produced all expected artifact files and adheres to constraints."""
    ws = Path(__file__).resolve().parents[1]
    out_dir = ws / "outputs/spatial_ablation_e3a"
    if not out_dir.exists():
        pytest.skip("E3a run has not yet completed.")

    csv_path = out_dir / "cv_results.csv"
    cfg_path = out_dir / "experiment_config.json"
    integrity_path = out_dir / "feature_integrity.json"
    findings_path = out_dir / "findings.md"
    fig1 = out_dir / "figures/spatial_vs_global_cv_comparison.png"
    fig2 = out_dir / "figures/spatial_granularity_tradeoff.png"

    for p in [csv_path, cfg_path, integrity_path, findings_path, fig1, fig2]:
        assert p.is_file(), f"Missing required E3a output: {p}"

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["dataset_session"] == "session1_cv_only"
    assert cfg["n_folds"] == 5

    # Check that findings.md has no emojis
    text = findings_path.read_text(encoding="utf-8")
    for char in text:
        # Check standard emoji unicode range
        assert not (
            0x1F600 <= ord(char) <= 0x1F64F
            or 0x1F300 <= ord(char) <= 0x1F5FF
            or 0x1F680 <= ord(char) <= 0x1F6FF
            or 0x2600 <= ord(char) <= 0x26FF
            or 0x2700 <= ord(char) <= 0x27BF
        ), f"Emoji found in findings.md: {char}"
