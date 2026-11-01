import numpy as np
from scipy.linalg import solve
from sklearn.svm import LinearSVC

from palmprint.evaluation.metrics import ranking
from palmprint.models.classical import (
    CollaborativeRidge,
    LinearSVM,
    NearestTemplate,
    model_definition,
)


def test_crc_matches_explicit_ridge():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(12, 7))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.repeat([1, 2, 3], 4)
    lam = 0.4
    query = rng.normal(size=(3, 7))
    model = CollaborativeRidge(lam).fit(X, labels)
    residual = model.scores(query)
    a = X.T
    expected = []
    for y in query:
        coef = solve(a.T @ a + lam * np.eye(len(labels)), a.T @ y)
        row = []
        for c in model.classes_:
            mask = labels == c
            row.append(np.linalg.norm(y - a[:, mask] @ coef[mask]))
        expected.append(row)
    assert np.allclose(residual, expected, atol=1e-6)


def test_all_classifiers_agree_on_ranking_contract():
    rng = np.random.default_rng(9)
    X = np.vstack([rng.normal(c * 3, 0.5, size=(6, 5)) for c in (1, 2, 3)])
    y = np.repeat([1, 2, 3], 6)
    for model in (NearestTemplate(), LinearSVM(), CollaborativeRidge()):
        model.fit(X, y)
        ranked, _ = ranking(model.scores(X), model.classes_)
        assert ranked.shape == (18, 3)
        assert np.all(np.sort(ranked, axis=1) == np.array([1, 2, 3]))


def test_crc_is_unnormalized_and_batch_independent():
    rng = np.random.default_rng(18)
    gallery = rng.normal(size=(15, 7))
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    # Noncontiguous, interleaved IDs exercise dictionary-to-class alignment.
    labels = np.tile([2, 17, 600], 5)
    query = rng.normal(size=(40, 7))
    model = CollaborativeRidge(0.4).fit(gallery, labels)
    a = gallery.T
    coef = solve(a.T @ a + 0.4 * np.eye(len(labels)), a.T @ query.T)
    residual = np.column_stack(
        [
            np.linalg.norm(query.T - a[:, labels == c] @ coef[labels == c], axis=0)
            for c in model.classes_
        ]
    )
    normalized = residual / np.column_stack(
        [np.linalg.norm(coef[labels == c], axis=0) for c in model.classes_]
    )
    assert np.allclose(model.scores(query, batch_size=1), residual, atol=1e-12)
    assert np.allclose(model.scores(query, batch_size=13), residual, atol=1e-12)
    assert not np.allclose(residual, normalized)
    assert np.any(residual.argmin(axis=1) != normalized.argmin(axis=1))
    assert np.array_equal(residual.argsort(axis=1), (residual**2).argsort(axis=1))
    assert model_definition(model)["coefficient_normalization"] == "none"


def test_svm_matches_liblinear_with_intercept_not_ridge():
    rng = np.random.default_rng(8)
    gallery = rng.normal(size=(30, 5)) + 3.0
    labels = np.tile([2, 17, 600], 10)
    query = rng.normal(size=(7, 5)) + 2.0
    model = LinearSVM(C=0.3).fit(gallery, labels)
    reference = LinearSVC(
        C=0.3,
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        fit_intercept=True,
        intercept_scaling=1.0,
        tol=1e-4,
        max_iter=30000,
        random_state=42,
        multi_class="ovr",
    ).fit(gallery, labels)
    assert np.array_equal(model.classes_, reference.classes_)
    assert np.allclose(model.scores(query), -reference.decision_function(query))
    assert np.allclose(-model.scores(query), query @ model.model.coef_.T + model.model.intercept_)
    assert np.any(np.abs(model.model.intercept_) > 0.01)
    targets = np.where(labels[:, None] == model.classes_[None, :], 1.0, -1.0)
    ridge_w = solve(gallery.T @ gallery + (1 / 0.3) * np.eye(5), gallery.T @ targets)
    assert not np.allclose(-model.scores(query), query @ ridge_w)
    assert model_definition(model)["parameters"]["fit_intercept"] is True
