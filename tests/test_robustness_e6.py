"""Unit and integration tests for E6 Robustness Analysis."""

from __future__ import annotations

import numpy as np

from scripts.run_robustness_e6 import (
    PERTURBATIONS,
    apply_blur3,
    apply_brightness_minus15,
    apply_brightness_plus15,
    apply_clean,
    apply_rotate_plus5,
    paired_cluster_bootstrap_diff,
)


def test_perturbation_functions_properties():
    # Create test synthetic image
    rng = np.random.default_rng(123)
    img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)

    # 1. Clean
    clean = apply_clean(img)
    assert np.array_equal(clean, img)
    assert clean.dtype == np.uint8

    # 2. Brightness minus 15
    b_minus = apply_brightness_minus15(img)
    assert b_minus.shape == (128, 128)
    assert b_minus.dtype == np.uint8
    assert b_minus.min() >= 0
    assert b_minus.max() <= 255
    # For pixels >= 15, should be exactly img - 15
    mask = img >= 15
    assert np.array_equal(b_minus[mask], img[mask] - 15)

    # 3. Brightness plus 15
    b_plus = apply_brightness_plus15(img)
    assert b_plus.shape == (128, 128)
    assert b_plus.dtype == np.uint8
    assert b_plus.min() >= 0
    assert b_plus.max() <= 255
    mask_high = img <= 240
    assert np.array_equal(b_plus[mask_high], img[mask_high] + 15)

    # 4. Blur 3x3
    blurred = apply_blur3(img)
    assert blurred.shape == (128, 128)
    assert blurred.dtype == np.uint8

    # 5. Rotate +5
    rotated = apply_rotate_plus5(img)
    assert rotated.shape == (128, 128)
    assert rotated.dtype == np.uint8


def test_perturbations_registry():
    assert set(PERTURBATIONS.keys()) == {
        "clean",
        "brightness_minus15",
        "brightness_plus15",
        "blur3",
        "rotate_plus5",
    }


def test_paired_cluster_bootstrap_identical():
    # If perturbed == clean, diff should be exactly 0
    palm_ids = np.repeat(np.arange(100), 10)
    correct = np.ones(1000, dtype=bool)

    res = paired_cluster_bootstrap_diff(correct, correct, palm_ids, n_resamples=1000, seed=42)
    assert res["observed_diff_pp"] == 0.0
    assert res["ci_95_low_pp"] == 0.0
    assert res["ci_95_high_pp"] == 0.0
    assert res["p_value"] == 1.0


def test_paired_cluster_bootstrap_degradation():
    rng = np.random.default_rng(42)
    palm_ids = np.repeat(np.arange(600), 10)

    # Clean is 98% accurate
    correct_clean = rng.binomial(1, 0.98, size=6000).astype(bool)
    # Perturbed is 70% accurate
    correct_pert = rng.binomial(1, 0.70, size=6000).astype(bool)

    res = paired_cluster_bootstrap_diff(
        correct_pert, correct_clean, palm_ids, n_resamples=2000, seed=42
    )
    # Diff should be around -28 pp
    assert res["observed_diff_pp"] < -20.0
    assert res["ci_95_low_pp"] < res["observed_diff_pp"]
    assert res["ci_95_high_pp"] > res["observed_diff_pp"]
    assert res["p_value"] < 0.01


def test_bootstrap_seed_determinism():
    palm_ids = np.repeat(np.arange(50), 4)
    rng = np.random.default_rng(99)
    c1 = rng.binomial(1, 0.9, size=200).astype(bool)
    c2 = rng.binomial(1, 0.8, size=200).astype(bool)

    res_a = paired_cluster_bootstrap_diff(c2, c1, palm_ids, n_resamples=500, seed=123)
    res_b = paired_cluster_bootstrap_diff(c2, c1, palm_ids, n_resamples=500, seed=123)

    assert res_a == res_b
