"""PCA implemented with NumPy linear algebra and randomized range finding.
"""
from __future__ import annotations

import numpy as np


class PCA:
    """Truncated PCA using randomized SVD of the centered training matrix.

    NumPy supplies matrix operations, QR and SVD; the PCA procedure itself is
    implemented here. Randomized SVD is an approximation to the leading axes.
    """

    def __init__(self, n_components: int, *, random_state: int = 42,
                 n_oversamples: int = 10, n_iter: int = 4):
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if n_oversamples < 0 or n_iter < 0:
            raise ValueError("n_oversamples and n_iter must be non-negative")
        self.n_components = n_components
        self.random_state = random_state
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter

    @staticmethod
    def _matrix(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            raise ValueError("X must be a two-dimensional real numeric matrix")
        x = x.astype(np.float32 if x.dtype == np.float32 else np.float64, copy=False)
        if not np.isfinite(x).all():
            raise ValueError("X must contain only finite values")
        return x

    def fit(self, x: np.ndarray) -> PCA:
        x = self._matrix(x)
        n_samples, n_features = x.shape
        if n_samples < 2 or self.n_components > min(n_samples, n_features):
            raise ValueError("need at least two samples and n_components <= min(X.shape)")
        mean = x.mean(axis=0)
        centered = x - mean
        # Random projection, then stabilized power iterations sharpen the spectrum.
        width = min(self.n_components + self.n_oversamples, n_samples, n_features)
        rng = np.random.default_rng(self.random_state)
        omega = rng.standard_normal((n_features, width)).astype(centered.dtype)
        q, _ = np.linalg.qr(centered @ omega, mode="reduced")
        for _ in range(self.n_iter):
            q_right, _ = np.linalg.qr(centered.T @ q, mode="reduced")
            q, _ = np.linalg.qr(centered @ q_right, mode="reduced")
        _, singular_values, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
        components = vt[:self.n_components].copy()
        # Fix the otherwise arbitrary sign of each singular vector.
        pivots = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(self.n_components), pivots])
        components *= np.where(signs == 0, 1, signs)[:, None]
        variance = singular_values[:self.n_components] ** 2 / (n_samples - 1)
        total_variance = np.einsum("ij,ij->", centered, centered, dtype=np.float64) / (n_samples - 1)
        self.mean_ = mean
        self.components_ = components
        self.singular_values_ = singular_values[:self.n_components]
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = (variance / total_variance if total_variance > 0
                                          else np.zeros_like(variance))
        self.n_features_in_ = n_features
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise ValueError("PCA must be fitted before transform")
        x = self._matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        return (x - self.mean_) @ self.components_.T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise ValueError("PCA must be fitted before inverse_transform")
        x = self._matrix(x)
        if x.shape[1] != self.n_components:
            raise ValueError("X has a different number of PCA components")
        return x @ self.components_ + self.mean_
