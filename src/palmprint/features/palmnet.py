"""Session-1-only variance selection for very high dimensional PalmNet CSR."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


class SparseVarianceSelector:
    """Select columns without a D x D covariance or a dense N x D matrix.

    Dimension is fixed before testing; this is a feature-selection experiment,
    not PCA compression and not a reproduction of the original descriptor.
    """

    def __init__(self, n_features=1024, batch_size=128):
        if n_features < 1 or batch_size < 1:
            raise ValueError("Selection dimensions and batch size must be positive.")
        self.n_features = int(n_features)
        self.batch_size = int(batch_size)

    def fit(self, X):
        if not sparse.issparse(X) or X.shape[0] < 2 or X.shape[1] < self.n_features:
            raise ValueError("Selection requires sparse training data with sufficient dimensions.")
        self.input_dimension_ = X.shape[1]
        sums = np.zeros(X.shape[1], dtype=np.float64)
        squares = np.zeros_like(sums)
        for start in range(0, X.shape[0], self.batch_size):
            sub = X[start : start + self.batch_size].tocsr()
            if not np.isfinite(sub.data).all():
                raise ValueError("Nonfinite feature values.")
            sums += np.bincount(sub.indices, weights=sub.data, minlength=X.shape[1])
            squares += np.bincount(sub.indices, weights=sub.data**2, minlength=X.shape[1])
        variance = np.maximum(squares / X.shape[0] - (sums / X.shape[0]) ** 2, 0)
        if np.count_nonzero(variance) < self.n_features:
            raise ValueError("Fewer varying features than requested selection dimensions.")
        # Threshold ties are resolved by ascending original column, deterministically.
        threshold = np.partition(variance, -self.n_features)[-self.n_features]
        above = np.flatnonzero(variance > threshold)
        tied = np.flatnonzero(variance == threshold)[: self.n_features - len(above)]
        self.columns_ = np.sort(np.r_[above, tied])
        self.training_images_ = X.shape[0]
        return self

    def transform(self, X):
        if not hasattr(self, "columns_") or X.shape[1] != self.input_dimension_:
            raise ValueError("Fit selector first and preserve input dimension.")
        result = np.empty((X.shape[0], len(self.columns_)), dtype=np.float64)
        for start in range(0, X.shape[0], self.batch_size):
            # Only the selected small matrix is densified, never raw PalmNet.
            sub = X[start : start + self.batch_size, self.columns_].toarray()
            result[start : start + len(sub)] = normalize(sub, norm="l2")
        return result
