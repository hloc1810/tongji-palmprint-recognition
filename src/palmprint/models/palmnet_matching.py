"""Exact sparse/dense partition of PalmNet dot products, without reducing features."""

import logging

import numpy as np
from scipy import sparse

from palmprint.training.palmnet import _validate_features

LOG = logging.getLogger(__name__)


def common_columns(gallery, count=65536, batch_size=64):
    """Computational partition based only on gallery column frequencies."""
    if count < 1:
        raise ValueError("Common-column count must be positive.")
    frequency = np.zeros(gallery.shape[1], dtype=np.int32)
    for i in range(0, gallery.shape[0], batch_size):
        frequency += np.bincount(
            gallery[i : i + batch_size].indices, minlength=gallery.shape[1]
        ).astype(np.int32)
    count = min(count, np.count_nonzero(frequency))
    if count == 0:
        return np.array([], dtype=np.int64)
    threshold = np.partition(frequency, -count)[-count]
    above = np.flatnonzero(frequency > threshold)
    tied = np.flatnonzero(frequency == threshold)[: count - len(above)]
    return np.sort(np.r_[above, tied])


def split_block(X, columns, mask):
    """Dense only a small selected subset; retain every other value in CSR."""
    dense = X[:, columns].toarray()
    keep = ~mask[X.indices]
    prefix = np.r_[0, np.cumsum(keep, dtype=np.int32)]
    remaining = sparse.csr_matrix((X.data[keep], X.indices[keep], prefix[X.indptr]), shape=X.shape)
    return dense, remaining


def hybrid_nearest_class_scores(
    probes,
    gallery,
    gallery_ids,
    *,
    batch_size=64,
    gallery_batch_size=256,
    common_count=65536,
    threads=4,
):
    """Same squared Euclidean as full sparse matching, using disjoint dot sums.

    q.g = q_common.g_common + q_remaining.g_remaining. The partition changes
    execution only: no feature is discarded, learned, projected or normalized.
    """
    _validate_features(probes)
    _validate_features(gallery)
    ids = np.asarray(gallery_ids)
    if (
        batch_size < 1
        or gallery_batch_size < 1
        or len(ids) != gallery.shape[0]
        or gallery.shape[0] == 0
        or probes.shape[1] != gallery.shape[1]
    ):
        raise ValueError("Invalid gallery labels/dimensions or batch sizes.")
    columns = common_columns(gallery, common_count)
    mask = np.zeros(gallery.shape[1], dtype=bool)
    mask[columns] = True
    LOG.info("Exact dot partition: %d dense columns + all remaining sparse columns", len(columns))
    classes = np.unique(ids)
    try:
        from sparse_dot_topn import sp_matmul
    except ImportError:
        sp_matmul = None

    def sparse_dot(a, b):
        if sp_matmul is None:
            return (a @ b).toarray()
        # Full multiplication API: no top-N truncation and no similarity threshold.
        return sp_matmul(a, b, n_threads=threads).toarray()

    result = np.full((probes.shape[0], len(classes)), np.inf)
    q2 = np.empty(probes.shape[0])
    for i in range(0, probes.shape[0], batch_size):
        q = probes[i : i + batch_size].astype(np.float64, copy=False)
        q2[i : i + q.shape[0]] = np.asarray(q.multiply(q).sum(axis=1)).ravel()
    for j in range(0, gallery.shape[0], gallery_batch_size):
        g = gallery[j : j + gallery_batch_size].astype(np.float64, copy=False)
        g2 = np.asarray(g.multiply(g).sum(axis=1)).ravel()
        gd, gs = split_block(g, columns, mask)
        transposed = gs.T.tocsr()
        block_ids = ids[j : j + g.shape[0]]
        block_classes = np.unique(block_ids)
        groups = [np.flatnonzero(block_ids == c) for c in block_classes]
        targets = np.searchsorted(classes, block_classes)
        for i in range(0, probes.shape[0], batch_size):
            q = probes[i : i + batch_size].astype(np.float64, copy=False)
            qd, qs = split_block(q, columns, mask)
            dots = qd @ gd.T + sparse_dot(qs, transposed)
            d2 = np.maximum(q2[i : i + q.shape[0], None] + g2[None, :] - 2 * dots, 0)
            minima = np.stack([d2[:, ix].min(axis=1) for ix in groups], axis=1)
            ix = np.ix_(np.arange(i, i + q.shape[0]), targets)
            result[ix] = np.minimum(result[ix], minima)
        LOG.info("Hybrid gallery matching: %d/%d", j + g.shape[0], gallery.shape[0])
        # Release the previous gallery block before the next allocation peak.
        del g, gd, gs, transposed
    return classes, result
