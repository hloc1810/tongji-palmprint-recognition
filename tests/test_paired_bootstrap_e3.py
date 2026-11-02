"""Unit and integration tests for E3: Paired Cluster Bootstrap Comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_cluster_bootstrap_synthetic(
    s_a: np.ndarray,
    s_b: np.ndarray,
    n_total: int,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Helper for synthetic paired cluster bootstrap."""
    rng = np.random.default_rng(seed)
    n_clusters = len(s_a)
    boot_idx = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    diffs = (s_a[boot_idx].sum(axis=1) - s_b[boot_idx].sum(axis=1)) / n_total * 100.0
    obs_diff = float((s_a.sum() - s_b.sum()) / n_total * 100.0)
    low, high = np.percentile(diffs, [2.5, 97.5])
    p_val = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return obs_diff, float(low), float(high), min(1.0, p_val)


def test_synthetic_paired_cluster_bootstrap_known_difference():
    """Verify paired cluster bootstrap on synthetic data with known deterministic difference."""
    n_clusters = 100
    probes_per_cluster = 10
    n_total = n_clusters * probes_per_cluster

    # Scenario 1: Method A has exactly 1 more correct probe per cluster than Method B (+10 pp)
    s_b = np.full(n_clusters, 6.0)  # 60%
    s_a = np.full(n_clusters, 7.0)  # 70%

    obs_diff, low, high, p_val = paired_cluster_bootstrap_synthetic(s_a, s_b, n_total)
    assert np.isclose(obs_diff, 10.0)
    # Since every cluster has exactly +1 difference, any resample sum is identical
    assert np.isclose(low, 10.0)
    assert np.isclose(high, 10.0)
    assert p_val == 0.0

    # Scenario 2: Identical methods -> diff is exactly 0
    obs_diff2, low2, high2, p_val2 = paired_cluster_bootstrap_synthetic(s_a, s_a, n_total)
    assert np.isclose(obs_diff2, 0.0)
    assert np.isclose(low2, 0.0)
    assert np.isclose(high2, 0.0)
    assert p_val2 == 1.0


def test_synthetic_paired_cluster_bootstrap_variable_clusters():
    """Verify bootstrap with stochastic cluster variance."""
    rng = np.random.default_rng(123)
    n_clusters = 200
    probes_per_cluster = 10
    n_total = n_clusters * probes_per_cluster

    # B has random scores around 7/10
    s_b = rng.binomial(probes_per_cluster, 0.70, size=n_clusters).astype(float)
    # A has random scores around 8.5/10
    s_a = rng.binomial(probes_per_cluster, 0.85, size=n_clusters).astype(float)

    obs_diff, low, high, p_val = paired_cluster_bootstrap_synthetic(
        s_a, s_b, n_total, n_resamples=5000, seed=42
    )

    # Observed diff should be around +15 pp
    assert 10.0 < obs_diff < 20.0
    # 95% CI should contain observed difference
    assert low < obs_diff < high
    # Significant difference
    assert p_val < 0.001


def test_feature_predictions_alignment_and_integrity():
    """Verify that predictions from raw, pca, and compcode align perfectly on Session 2."""
    ws = Path(__file__).resolve().parents[1]
    pred_dir = ws / "outputs/feature_ablation/predictions"

    df_raw = pd.read_csv(pred_dir / "raw_predictions.csv")
    df_pca = pd.read_csv(pred_dir / "pca_predictions.csv")
    df_comp = pd.read_csv(pred_dir / "compcode_predictions.csv")

    assert len(df_raw) == 6000
    assert len(df_pca) == 6000
    assert len(df_comp) == 6000

    assert df_raw.sample_uid.is_unique
    assert df_pca.sample_uid.is_unique
    assert df_comp.sample_uid.is_unique

    # Sort each by sample_uid and assert true_palm_id match
    df_raw_sorted = df_raw.sort_values("sample_uid").reset_index(drop=True)
    df_pca_sorted = df_pca.sort_values("sample_uid").reset_index(drop=True)
    df_comp_sorted = df_comp.sort_values("sample_uid").reset_index(drop=True)

    np.testing.assert_array_equal(df_raw_sorted.sample_uid, df_comp_sorted.sample_uid)
    np.testing.assert_array_equal(df_pca_sorted.sample_uid, df_comp_sorted.sample_uid)
    np.testing.assert_array_equal(df_raw_sorted.true_palm_id, df_comp_sorted.true_palm_id)
    np.testing.assert_array_equal(df_pca_sorted.true_palm_id, df_comp_sorted.true_palm_id)


def test_paired_bootstrap_artifacts_and_no_emojis():
    """Verify that E3 bootstrap output files exist, match schema, and have no emojis."""
    ws = Path(__file__).resolve().parents[1]
    feat_dir = ws / "outputs/feature_ablation"

    res_csv = feat_dir / "paired_bootstrap_results.csv"
    cfg_json = feat_dir / "bootstrap_config.json"
    fig_png = feat_dir / "figures/paired_feature_differences.png"
    findings_md = feat_dir / "findings.md"

    for p in [res_csv, cfg_json, fig_png, findings_md]:
        assert p.is_file(), f"Missing required artifact: {p}"

    # Verify CSV schema
    df = pd.read_csv(res_csv)
    assert len(df) == 3
    assert set(df["comparison"]) == {"CompCode - Raw", "CompCode - PCA", "Raw - PCA"}
    expected_cols = {
        "comparison",
        "method_a",
        "method_b",
        "n_samples",
        "n_clusters",
        "rank1_a_pct",
        "rank1_b_pct",
        "observed_diff_pp",
        "ci_95_low_pp",
        "ci_95_high_pp",
        "bootstrap_mean_pp",
        "bootstrap_std_pp",
        "p_value",
    }
    assert expected_cols.issubset(set(df.columns))

    # Verify config JSON
    with open(cfg_json, encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["bootstrap_iterations"] == 10000
    assert cfg["random_seed"] == 42
    assert cfg["n_samples"] == 6000
    assert cfg["n_clusters"] == 600

    # Verify no emojis in findings.md
    text = findings_md.read_text(encoding="utf-8")
    assert "## 5. So sánh Thống kê Cặp đôi" in text
    for char in text:
        assert not (
            0x1F600 <= ord(char) <= 0x1F64F
            or 0x1F300 <= ord(char) <= 0x1F5FF
            or 0x1F680 <= ord(char) <= 0x1F6FF
            or 0x2600 <= ord(char) <= 0x26FF
            or 0x2700 <= ord(char) <= 0x27BF
        ), f"Emoji found in findings.md: {char}"
