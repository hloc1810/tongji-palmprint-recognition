"""Compare the mean accuracy of fixed CNN seeds with a paired classical baseline."""
from __future__ import annotations

import numpy as np


def paired_seed_mean_delta(y, seed_predictions, baseline_predictions, *,
                           iterations: int = 5000, seed: int = 42) -> dict:
    """Palm-cluster bootstrap; the seed mean is a metric, not an ensemble predictor.

    Each replicate resamples Palm IDs once and applies the same probes to every
    seed and the baseline. This CI conditions on the supplied seeds. Seed sample
    standard deviation is reported separately; probes are never pooled as 3N.
    """
    y = np.asarray(y)
    predictions = np.asarray(seed_predictions)
    baseline = np.asarray(baseline_predictions)
    if (y.ndim != 1 or not len(y) or predictions.ndim != 2
            or predictions.shape[0] < 2 or predictions.shape[1] != len(y)
            or baseline.shape != y.shape or iterations < 100):
        raise ValueError("Need >=2 seeds aligned with the same nonempty probe set")
    delta = (predictions == y[None, :]).mean(axis=0) - (baseline == y)
    palms = np.unique(y)
    sums = np.array([delta[y == palm].sum() for palm in palms])
    counts = np.array([(y == palm).sum() for palm in palms])
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(palms), size=(iterations, len(palms)))
    samples = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    # Three primary baseline comparisons. Bonferroni intervals are provided in
    # addition to ordinary descriptive 95% intervals; no nine-test seed picking.
    low, high = np.quantile(samples, [.025, .975])
    family_low, family_high = np.quantile(samples, [.05 / 6, 1 - .05 / 6])
    return {"delta_rank1_cnn_mean_minus_classical": float(delta.mean()),
            "ci95_lower": float(low), "ci95_upper": float(high),
            "familywise95_bonferroni3_lower": float(family_low),
            "familywise95_bonferroni3_upper": float(family_high),
            "n_probes": len(y), "n_seeds": predictions.shape[0],
            "n_palm_clusters": len(palms), "iterations": iterations,
            "bootstrap_seed": seed, "conditions_on_fixed_seeds": True}
