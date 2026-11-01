import numpy as np
import pytest

from palmprint.evaluation.metrics import bootstrap_delta, bootstrap_rank1, compute_metrics, ranking


def test_rank1_rank5_classwise():
    labels = np.arange(1, 7)
    scores = np.array([[2, 1, 0, 4, 3, 6], [0, 1, 2, 3, 4, 10]])
    order, _ = ranking(scores, labels)
    result = compute_metrics(np.array([3, 6]), order, labels)
    assert result["rank1"] == 0.5
    assert result["rank5"] == 0.5
    assert len(set(order[0, :5])) == 5


def test_stable_tie_and_bad_data():
    ids, _ = ranking(np.zeros((1, 3)), np.array([1, 3, 9]))
    assert ids.tolist() == [[1, 3, 9]]
    with pytest.raises(ValueError):
        ranking(np.array([[float("nan"), 1, 2]]), np.array([1, 2, 3]))


def test_bootstrap_reproducible():
    y = np.repeat([1, 2, 3], 4)
    a = y.copy()
    a[0] = 2
    b = y.copy()
    b[3:5] = 3
    assert bootstrap_rank1(y, a) == bootstrap_rank1(y, a)
    ci = bootstrap_delta(y, a, b)
    assert ci[0] <= ci[1]
