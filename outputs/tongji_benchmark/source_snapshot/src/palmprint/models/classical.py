"""Classwise-ranking models on the same L2-normalized CompCode vectors."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.svm import LinearSVC


class NearestTemplate:
    """1-NN, with each class scored by its closest gallery template."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestTemplate":
        self.gallery = np.asarray(X, dtype=np.float32)
        self.labels = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(self.labels)
        self.sq_norm = (self.gallery ** 2).sum(axis=1)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        probes = np.atleast_2d(np.asarray(X, dtype=np.float32))
        squared = np.maximum(0, (probes**2).sum(axis=1, keepdims=True)
                             + self.sq_norm[None, :] - 2 * probes @ self.gallery.T)
        return np.stack([squared[:, self.labels == c].min(axis=1)
                         for c in self.classes_], axis=1)


class LinearSVM:
    """Liblinear L2 squared-hinge OvR SVM with a penalized synthetic intercept."""
    def __init__(self, C: float = 1.0, max_iter: int = 30000):
        self.C = float(C)
        self.max_iter = int(max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        self.model = LinearSVC(C=self.C, penalty="l2", loss="squared_hinge", dual=False,
                               fit_intercept=True, intercept_scaling=1.0, tol=1e-4,
                               max_iter=self.max_iter, random_state=42,
                               multi_class="ovr")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        if len(self.classes_) < 3:
            raise ValueError("Use >=3 classes for unambiguous classwise SVM scores")
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(self.model.decision_function(X))
        return -values  # universal convention: lower score is better (not probability)


class CollaborativeRidge:
    """CRC_RLS: dictionary-wide ridge coefficients and per-class reconstruction residual.

    Uses Cholesky on the feature-space Gram matrix (m x m); avoids an n x n inverse.
    Residual uses Gram identities, avoiding constructing an m-vector per palm per probe.
    Scores are unnormalized Euclidean residuals, NOT divided by ||coeff_class||.
    Tongji Eq. (6) uses their squares, which have the same class ranking.
    """
    def __init__(self, lam: float = 1.35):
        if lam <= 0:
            raise ValueError("lambda must be positive")
        self.lam = float(lam)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CollaborativeRidge":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        self.A = X.T.copy()  # m x n
        self.labels = y.copy()
        m = X.shape[1]
        gram = self.A @ self.A.T
        gram.flat[::m + 1] += self.lam
        self.factor = cho_factor(gram, lower=True, overwrite_a=True,
                                 check_finite=True)
        self.class_indices = [np.flatnonzero(y == c) for c in self.classes_]
        self.class_grams = [self.A[:, ix].T @ self.A[:, ix]
                            for ix in self.class_indices]
        return self

    def scores(self, X: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        result = np.empty((len(X), len(self.classes_)), dtype=np.float64)
        for start in range(0, len(X), batch_size):
            Y = X[start:start+batch_size].T  # m x b
            coeff = self.A.T @ cho_solve(self.factor, Y, check_finite=False)  # n x b
            aty = self.A.T @ Y
            y2 = np.sum(Y*Y, axis=0)
            for c, (idx, gram) in enumerate(zip(self.class_indices, self.class_grams)):
                x = coeff[idx]
                # ||y - A_c*x_c||^2 = ||y||^2 + x'G_c x - 2 x'A_c'y.
                residual_sq = y2 + np.sum(x * (gram @ x), axis=0) - 2 * np.sum(x * aty[idx], axis=0)
                result[start:start+Y.shape[1], c] = np.sqrt(np.maximum(residual_sq, 0))
        return result


def build_model(name: str, parameter: float):
    if name == "knn":
        if parameter != 1:
            raise ValueError("NearestTemplate is strictly 1-NN")
        return NearestTemplate()
    if name == "svm":
        return LinearSVM(C=parameter)
    if name == "crc":
        return CollaborativeRidge(lam=parameter)
    raise ValueError(f"Unknown model: {name}")


def model_definition(model) -> dict:
    """Record the actual fitted variant so reports cannot confuse ridge and SVM."""
    if isinstance(model, NearestTemplate):
        return {"method": "1-NN nearest-template", "score_kind": "squared_euclidean",
                "score_formula": "min_j_in_class ||query-template_j||_2^2"}
    if isinstance(model, LinearSVM):
        return {"method": "LinearSVC L2 squared-hinge OvR with intercept",
                "score_kind": "negative_svm_margin",
                "score_formula": "-(w_class.T @ query + intercept_class)",
                "parameters": model.model.get_params(deep=False)}
    if isinstance(model, CollaborativeRidge):
        return {"method": "CRC ridge, unnormalized Euclidean residual",
                "score_kind": "reconstruction_residual",
                "score_formula": "||query-A_class @ alpha_class||_2",
                "coefficient_normalization": "none", "lambda": model.lam,
                "paper_relation": "sqrt of Tongji 2017 Eq. (6); identical ranking"}
    raise TypeError(f"Unsupported model: {type(model).__name__}")
