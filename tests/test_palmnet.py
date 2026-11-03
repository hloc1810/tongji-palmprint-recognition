"""Independent distance oracle, memory-safe cache, split and reduction regressions."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial.distance import cdist

from palmprint.features.palmnet import SparseVarianceSelector
from palmprint.training.palmnet import extract_cached, nearest_class_scores, paths_for


@pytest.mark.parametrize("query_batch,gallery_batch", [(1, 1), (2, 3), (9, 20)])
def test_sparse_matching_agrees_with_dense_oracle(query_batch, gallery_batch):
    rng = np.random.default_rng(7)
    g = rng.integers(0, 4, (9, 13)).astype(float)
    q = rng.integers(0, 4, (5, 13)).astype(float)
    labels = np.array([7, 3, 42, 3, 7, 42, 7, 3, 42])
    classes, scores = nearest_class_scores(
        sparse.csr_matrix(q),
        sparse.csr_matrix(g),
        labels,
        batch_size=query_batch,
        gallery_batch_size=gallery_batch,
    )
    distances = cdist(q, g, metric="sqeuclidean")
    expected = np.stack([distances[:, labels == c].min(axis=1) for c in classes], axis=1)
    np.testing.assert_allclose(scores, expected, atol=1e-10)
    np.testing.assert_array_equal(classes, [3, 7, 42])


@pytest.mark.parametrize("invalid", ["nan", "dimension", "labels", "batch", "empty"])
def test_matching_rejects_invalid_input(invalid):
    q, g, ids, batch = sparse.eye(3, format="csr"), sparse.eye(3, format="csr"), [1, 2, 3], 2
    if invalid == "nan":
        q.data[0] = np.nan
    elif invalid == "dimension":
        q = sparse.eye(4, format="csr")
    elif invalid == "labels":
        ids = [1]
    elif invalid == "batch":
        batch = 0
    else:
        g, ids = g[:0], []
    with pytest.raises(ValueError):
        nearest_class_scores(q, g, ids, batch_size=batch)


class FakeExtractor:
    feature_dimension = 8

    def __init__(self):
        self.calls = 0

    def transform(self, paths, batch_size):
        self.calls += 1
        return sparse.csr_matrix([[int(p) + 1, 0, 0, 1, 0, 0, 0, 0] for p in paths])


def test_cache_resume_order_model_and_corruption(tmp_path):
    data = pd.DataFrame({"sample_uid": ["a", "b", "c"], "sha256": ["aa", "bb", "cc"]})
    model = FakeExtractor()
    X = extract_cached(model, [0, 1, 2], data, tmp_path, "model1", chunk_size=2)
    expected = np.array(
        [[1, 0, 0, 1, 0, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0, 0], [3, 0, 0, 1, 0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(X.toarray(), expected)
    assert model.calls == 2
    assert not X.data.flags.writeable
    again = extract_cached(model, [0, 1, 2], data, tmp_path, "model1", chunk_size=2)
    np.testing.assert_array_equal(again.toarray(), expected)
    assert model.calls == 2
    for changed, model_sha in [(data.iloc[::-1], "model1"), (data, "model2")]:
        with pytest.raises(RuntimeError, match="another model/order"):
            extract_cached(model, [0, 1, 2], changed, tmp_path, model_sha, chunk_size=2)
    with (tmp_path / "data.bin").open("r+b") as file:
        file.write(b"corrupt!")
    with pytest.raises(RuntimeError, match="modified"):
        extract_cached(model, [0, 1, 2], data, tmp_path, "model1", chunk_size=2)


def test_selection_matches_variance_and_transform_never_refits():
    train = sparse.csr_matrix([[0.0, 1, 0, 1], [8, 1, 0, 2], [0, 1, 8, 3]])
    selector = SparseVarianceSelector(n_features=2, batch_size=2).fit(train)
    np.testing.assert_array_equal(selector.columns_, [0, 2])
    before = selector.columns_.copy()
    probe = sparse.csr_matrix([[0.0, 999, 2, 999]])
    np.testing.assert_allclose(selector.transform(probe), [[0, 1]])
    np.testing.assert_array_equal(selector.columns_, before)
    np.testing.assert_allclose(selector.transform(train)[1], [1, 0])


def test_paths_only_decode_requested_session(tmp_path, monkeypatch):
    import palmprint.training.palmnet as m5

    calls = []
    monkeypatch.setattr(m5, "read_gray", lambda path, size: calls.append(path))
    monkeypatch.setattr(m5, "hash_file", lambda path: "ok")
    data = pd.DataFrame(
        {
            "session": ["session2", "session1"],
            "palm_id": [1, 1],
            "path": ["session2/b.bmp", "session1/a.bmp"],
            "sample_uid": ["session2/b", "session1/a"],
            "sha256": ["ok", "ok"],
        }
    )
    part, paths = paths_for(data, tmp_path, "session1")
    assert part.sample_uid.tolist() == ["session1/a"]
    assert calls == paths == [tmp_path / "session1/a.bmp"]
    data.loc[1, "path"] = "../outside.bmp"
    with pytest.raises(ValueError, match="outside dataset root"):
        paths_for(data, tmp_path, "session1")


def test_classifier_fit_is_blocked_after_probe_extraction(tmp_path):
    from argparse import Namespace

    from palmprint.training.palmnet_classifiers import train_classifiers

    cache = tmp_path / "outputs/palmnet_m5/cache/probe"
    cache.mkdir(parents=True)
    (cache / "index.json").write_text("{}")
    with pytest.raises(RuntimeError, match="before accessing Session2"):
        train_classifiers(Namespace(workspace=str(tmp_path), palmnet_python=None))


@pytest.mark.parametrize("common_count", [1, 4, 20])
def test_exact_partition_preserves_every_distance_and_class_minimum(common_count):
    from palmprint.models.palmnet_matching import hybrid_nearest_class_scores

    rng = np.random.default_rng(123)
    gallery = rng.normal(size=(11, 20))
    probe = rng.normal(size=(7, 20))
    gallery[rng.random(gallery.shape) < 0.7] = 0
    probe[rng.random(probe.shape) < 0.7] = 0
    labels = np.array([3, 9, 1, 3, 9, 1, 3, 9, 1, 3, 9])
    classes, scores = hybrid_nearest_class_scores(
        sparse.csr_matrix(probe),
        sparse.csr_matrix(gallery),
        labels,
        batch_size=3,
        gallery_batch_size=4,
        common_count=common_count,
    )
    distances = cdist(probe, gallery, metric="sqeuclidean")
    expected = np.stack([distances[:, labels == c].min(axis=1) for c in classes], axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
