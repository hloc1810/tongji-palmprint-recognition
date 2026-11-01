import numpy as np
import pytest

from palmprint.reporting.cnn_comparison import paired_seed_mean_delta


def test_seed_mean_is_not_pooled_or_ensemble_accuracy():
    y = np.array([1, 1, 2, 2, 3, 3])
    wrong = np.array([2, 2, 3, 3, 1, 1])
    # Two perfect runs and one entirely wrong run: mean is 2/3, even though
    # majority voting would produce a perfect ensemble on these probes.
    result = paired_seed_mean_delta(y, np.stack([y, y, wrong]), wrong)
    assert result["delta_rank1_cnn_mean_minus_classical"] == pytest.approx(2 / 3)
    assert result["ci95_lower"] == pytest.approx(2 / 3)
    assert result["ci95_upper"] == pytest.approx(2 / 3)
    assert result["n_probes"] == 6
    assert result["n_palm_clusters"] == 3
    assert result["n_seeds"] == 3


def test_identical_predictions_and_shape_errors():
    y = np.array([1, 1, 2, 2])
    result = paired_seed_mean_delta(y, np.stack([y, y, y]), y)
    assert result["delta_rank1_cnn_mean_minus_classical"] == 0
    assert result["ci95_lower"] == result["ci95_upper"] == 0
    with pytest.raises(ValueError):
        paired_seed_mean_delta(y, np.stack([y]), y)
    with pytest.raises(ValueError):
        paired_seed_mean_delta(y, np.ones((3, 5)), y)
