from __future__ import annotations

import numpy as np
import pytest

from palmprint.features.pca import PCA


def test_pca_matches_exact_svd_and_reconstructs():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 8)) @ np.diag([9, 7, 5, 4, 3, 2, 1, 0.5]) + 3
    pca = PCA(4, random_state=42).fit(x)
    centered = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    np.testing.assert_allclose(pca.components_.T @ pca.components_, vt[:4].T @ vt[:4], atol=1e-10)
    np.testing.assert_allclose(pca.explained_variance_ratio_, s[:4] ** 2 / (s ** 2).sum())
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(x)),
                               centered @ vt[:4].T @ vt[:4] + x.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(4), atol=1e-10)


def test_pca_randomized_reproducible_and_approximate():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(100, 60)) @ np.diag(np.geomspace(10, 0.01, 60))
    a = PCA(5, random_state=3).fit(x)
    b = PCA(5, random_state=3).fit(x)
    np.testing.assert_array_equal(a.components_, b.components_)
    _, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(a.singular_values_, s[:5], rtol=1e-4)
    np.testing.assert_allclose(a.fit_transform(x), a.transform(x))


def test_pca_validates_input_and_constant_data():
    with pytest.raises(ValueError):
        PCA(2).transform(np.ones((2, 2)))
    with pytest.raises(ValueError):
        PCA(3).fit(np.ones((2, 4)))
    with pytest.raises(ValueError):
        PCA(1).fit(np.array([[1.0], [np.nan]]))
    model = PCA(2).fit(np.ones((3, 4)))
    np.testing.assert_array_equal(model.explained_variance_ratio_, np.zeros(2))
    np.testing.assert_array_equal(model.transform(np.ones((1, 4))), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        model.transform(np.ones((1, 3)))
