"""M5 PCA-Gabor PalmNet: author Python port, Session1 -> Session2, sparse IO."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from palmprint.data.dataset import read_gray, resolve_path, validate_manifest
from palmprint.evaluation.metrics import bootstrap_rank1, compute_metrics, ranking
from palmprint.utils import atomic_json, hash_file, hash_object

LOG = logging.getLogger(__name__)
VARIANT = "PCA-Gabor PalmNet (author Python port; not all paper variants)"
SOURCE_URL = "https://github.com/AngeloUNIMI/PalmNet"


def read_context(ws, data_root_arg=None):
    manifest_path = ws / "data/metadata/manifest.csv"
    frame = pd.read_csv(manifest_path)
    required = {"session", "palm_id", "path", "sample_uid", "sha256", "fold", "height", "width"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame.columns))}")
    if frame.sample_uid.duplicated().any() or frame.path.duplicated().any():
        raise ValueError("Manifest contains duplicate UID or file paths.")
    validate_manifest(frame, folds=5, strict=True)
    for session in ("session1", "session2"):
        counts = frame[frame.session == session].groupby("palm_id").size()
        if set(counts.index) != set(range(1, 601)) or not (counts == 10).all():
            raise ValueError(f"{session}: expected 600 Palm IDs x 10 images.")
    if not ((frame.height == 128) & (frame.width == 128)).all():
        raise ValueError("M5 requires audited Official ROI 128x128.")
    if data_root_arg:
        root = Path(data_root_arg).expanduser().resolve()
    elif (ws / "data/raw/session1").is_dir() and (ws / "data/raw/session2").is_dir():
        root = (ws / "data/raw").resolve()
    else:
        info = json.loads((ws / "data/metadata/local_dataset.json").read_text(encoding="utf-8"))
        root = Path(info["data_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ROI root unavailable: {root}. Supply --data-root.")
    return frame, root, hash_file(manifest_path)


def paths_for(frame, root, session, verify_hash=True):
    part = (
        frame.loc[frame.session == session]
        .sort_values(["palm_id", "sample_uid"], kind="stable")
        .reset_index(drop=True)
    )
    paths = []
    for row in part.itertuples(index=False):
        path = resolve_path(root, str(row.path))
        if verify_hash and hash_file(path) != str(row.sha256):
            raise ValueError(f"ROI sha256 mismatch: {row.sample_uid}")
        read_gray(path, 128)
        paths.append(path)
    return part, paths


def get_vendor(vendor_path):
    vendor = Path(vendor_path).expanduser().resolve()
    if not (vendor / "palmnet/model.py").is_file():
        raise FileNotFoundError(f"Expected author source at {vendor}/palmnet/model.py")
    hashes = {
        p.relative_to(vendor).as_posix(): hash_file(p)
        for p in sorted((vendor / "palmnet").rglob("*.py"))
    }
    info = {"url": SOURCE_URL, "source_sha256": hash_object(hashes), "files": hashes}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = None
    existing = sys.modules.get("palmnet")
    if existing and not Path(existing.__file__).resolve().is_relative_to(vendor):
        raise RuntimeError("Another palmnet is imported; use a fresh process.")
    sys.path.insert(0, str(vendor))
    return (
        importlib.import_module("palmnet.config").PalmConfig,
        importlib.import_module("palmnet.model").PalmPCAGabor,
        info,
    )


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def _validate_features(X):
    if not sparse.issparse(X) or X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Expected a sparse descriptor matrix.")
    for start in range(0, len(X.data), 1_000_000):
        if not np.isfinite(X.data[start : start + 1_000_000]).all():
            raise ValueError("Nonfinite PalmNet descriptor.")


def nearest_class_scores(probes, gallery, gallery_ids, *, batch_size=48, gallery_batch_size=256):
    """Exact min-template squared L2; bounded RAM for disk-backed CSR.

    Only the pairwise distance blocks and final class scores become dense.
    Labels may be ungrouped; minima are merged across all gallery blocks.
    """
    if batch_size < 1 or gallery_batch_size < 1:
        raise ValueError("Matching batch sizes must be positive.")
    _validate_features(probes)
    _validate_features(gallery)
    ids = np.asarray(gallery_ids)
    if (
        gallery.shape[0] == 0
        or ids.ndim != 1
        or len(ids) != gallery.shape[0]
        or probes.shape[1] != gallery.shape[1]
    ):
        raise ValueError("Mismatched feature dimensions or gallery length.")
    classes = np.unique(ids)
    result = np.full((probes.shape[0], len(classes)), np.inf, dtype=np.float64)
    q2 = np.empty(probes.shape[0], dtype=np.float64)
    for i in range(0, probes.shape[0], batch_size):
        sub = probes[i : i + batch_size].astype(np.float64, copy=False)
        q2[i : i + sub.shape[0]] = np.asarray(sub.multiply(sub).sum(axis=1)).ravel()
    for j in range(0, gallery.shape[0], gallery_batch_size):
        g = gallery[j : j + gallery_batch_size].astype(np.float64, copy=False)
        g2 = np.asarray(g.multiply(g).sum(axis=1)).ravel()
        transposed = g.T.tocsr()
        block_ids = ids[j : j + g.shape[0]]
        block_classes = np.unique(block_ids)
        columns = [np.flatnonzero(block_ids == c) for c in block_classes]
        target = np.searchsorted(classes, block_classes)
        for i in range(0, probes.shape[0], batch_size):
            q = probes[i : i + batch_size].astype(np.float64, copy=False)
            dots = (q @ transposed).toarray()
            d2 = np.maximum(q2[i : i + q.shape[0], None] + g2[None, :] - 2 * dots, 0)
            minima = np.stack([d2[:, ix].min(axis=1) for ix in columns], axis=1)
            ix = np.ix_(np.arange(i, i + q.shape[0]), target)
            result[ix] = np.minimum(result[ix], minima)
        LOG.info("Gallery matching: %d/%d", j + g.shape[0], gallery.shape[0])
    return classes, result


def _cache_signature(model_sha, data):
    return hash_object(
        {"model_sha256": model_sha, "samples": data[["sample_uid", "sha256"]].to_dict("records")}
    )


def extract_cached(model, paths, data, cache, model_sha, *, batch_size=2, chunk_size=64):
    """Hash-checked sparse shards, resumable without refitting or changing order."""
    cache.mkdir(parents=True, exist_ok=True)
    signature = _cache_signature(model_sha, data)
    index_path = cache / "index.json"
    index = {
        "signature": signature,
        "chunks": {},
        "rows": len(paths),
        "columns": model.feature_dimension,
        "chunk_size": chunk_size,
    }
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["signature"] != signature or index["chunk_size"] != chunk_size:
            raise RuntimeError("Feature cache belongs to another model/order/chunk size.")
    for start in range(0, len(paths), chunk_size):
        key = f"{start:06d}"
        dest = cache / f"{key}.npz"
        if key in index["chunks"]:
            if hash_file(dest) != index["chunks"][key]:
                raise RuntimeError(f"Feature cache modified: {dest}")
            continue
        X = model.transform(paths[start : start + chunk_size], batch_size=batch_size)
        _validate_features(X)
        if X.shape != (min(chunk_size, len(paths) - start), model.feature_dimension):
            raise RuntimeError("Wrong PalmNet output shape.")
        tmp = cache / f"{key}.tmp.npz"
        sparse.save_npz(tmp, X, compressed=True)
        tmp.replace(dest)
        index["chunks"][key] = hash_file(dest)
        atomic_json(index_path, index)
        LOG.info("Saved sparse descriptors: %d/%d", start + X.shape[0], len(paths))
    return assemble_csr(cache, index)


def assemble_csr(cache, index):
    """Assemble on disk, holding at most one shard in RAM; never sparse.vstack."""
    meta_path = cache / "csr.json"
    signature = hash_object(index)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["signature"] != signature:
            raise RuntimeError("CSR cache signature mismatch.")
        for name, digest in meta["hashes"].items():
            if hash_file(cache / name) != digest:
                raise RuntimeError(f"CSR cache modified: {name}")
        return load_csr(cache)
    pointers = [0]
    with (cache / "data.bin").open("wb") as values, (cache / "indices.bin").open("wb") as cols:
        for key in sorted(index["chunks"]):
            X = sparse.load_npz(cache / f"{key}.npz").tocsr()
            X.data.astype(np.float64, copy=False).tofile(values)
            X.indices.astype(np.int32, copy=False).tofile(cols)
            pointers.extend((X.indptr[1:].astype(np.int64) + pointers[-1]).tolist())
    ptr = np.asarray(pointers, dtype=np.int64)
    np.save(cache / "indptr.npy", ptr, allow_pickle=False)
    meta = {
        "signature": signature,
        "shape": [index["rows"], index["columns"]],
        "nnz": int(ptr[-1]),
        "hashes": {
            name: hash_file(cache / name) for name in ("data.bin", "indices.bin", "indptr.npy")
        },
    }
    atomic_json(meta_path, meta)
    return load_csr(cache)


def load_csr(cache):
    meta = json.loads((cache / "csr.json").read_text())
    if meta["nnz"] >= np.iinfo(np.int32).max:
        raise ValueError("CSR nnz exceeds int32 capacity.")
    data = np.memmap(cache / "data.bin", mode="r", dtype=np.float64, shape=(meta["nnz"],))
    indices = np.memmap(cache / "indices.bin", mode="r", dtype=np.int32, shape=(meta["nnz"],))
    ptr = np.load(cache / "indptr.npy", allow_pickle=False).astype(np.int32)
    return sparse.csr_matrix((data, indices, ptr), shape=tuple(meta["shape"]), copy=False)


def _paths(args):
    ws = Path(args.workspace).resolve()
    vendor = Path(args.palmnet_python or ws / ".runtime/PalmNet/python").resolve()
    return ws, ws / "artifacts/palmnet_m5", ws / "outputs/palmnet_m5", vendor


def _check_vendor(frozen, info):
    if frozen["vendor"]["source_sha256"] != info["source_sha256"]:
        raise RuntimeError("PalmNet source changed since filter training.")


def _check_frozen(dest, manifest_sha, info):
    frozen = json.loads((dest / "freeze.json").read_text())
    if manifest_sha != frozen["manifest_sha256"]:
        raise RuntimeError("Manifest changed after training.")
    _check_vendor(frozen, info)
    for name, digest in frozen["artifact_hashes"].items():
        if hash_file(dest / name) != digest:
            raise RuntimeError(f"Frozen artifact modified: {name}")
    return frozen


def train(args):
    ws, dest, out, vendor = _paths(args)
    frame, root, manifest_sha = read_context(ws, args.data_root)
    data, paths = paths_for(frame, root, "session1", verify_hash=not args.skip_sha)
    PalmConfig, PalmPCAGabor, info = get_vendor(vendor)
    if (dest / "freeze.json").exists():
        raise FileExistsError("M5 is frozen. Use evaluate; do not overwrite.")
    dest.mkdir(parents=True, exist_ok=True)
    cfg = PalmConfig(
        image_size=128,
        pca_patch_size=15,
        pca_filters=args.pca_filters,
        default_orientations=args.fixed_orientations,
        adaptive_orientations=args.adaptive_orientations,
        selected_wavelets=args.selected_wavelets,
        hist_block_size=(args.hist_block, args.hist_block),
        block_overlap=0,
        dtype=args.dtype,
        execution_backend=args.backend,
        convolution_backend=args.convolution,
        response_batch_size=args.response_batch,
        pca_image_batch_size=args.pca_image_batch,
        orientation_batch_size=args.orientation_batch,
        gabor_tuning_batch_size=args.gabor_tuning_batch,
        max_patches_per_image=args.max_patches_per_image,
        num_workers=args.workers,
        seed=args.seed,
    ).validate()
    fit_path = dest / "fit.json"
    if fit_path.exists():
        fitted = json.loads(fit_path.read_text())
        _check_vendor(fitted, info)
        if (
            fitted["manifest_sha256"] != manifest_sha
            or hash_object(fitted["config"]) != hash_object(cfg.to_dict())
            or hash_file(dest / "model.pt") != fitted["model_sha256"]
        ):
            raise RuntimeError("Cannot resume changed model/config/manifest.")
        model = PalmPCAGabor.load(dest / "model.pt", args.device)
    else:
        model = PalmPCAGabor(cfg, args.device)
        start = time.perf_counter()
        model.fit(paths)
        elapsed = time.perf_counter() - start
        model.save(dest / "model.pt")
        import torch

        fitted = {
            "variant": VARIANT,
            "manifest_sha256": manifest_sha,
            "model_sha256": hash_file(dest / "model.pt"),
            "vendor": info,
            "config": cfg.to_dict(),
            "fit_seconds": elapsed,
            "fit_images": len(paths),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(model.device),
                "threads": torch.get_num_threads(),
            },
            "hash_verification": not args.skip_sha,
            "data_root": str(root),
        }
        atomic_json(fit_path, fitted)
    start = time.perf_counter()
    gallery = extract_cached(
        model,
        paths,
        data,
        out / "cache/gallery",
        fitted["model_sha256"],
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    np.save(dest / "gallery_ids.npy", data.palm_id.to_numpy(dtype=np.int32), allow_pickle=False)
    atomic_json(dest / "gallery_uids.json", data.sample_uid.tolist())
    frozen = dict(
        fitted,
        status="frozen_session1_only",
        feature_dim=int(gallery.shape[1]),
        gallery_nonzeros=int(gallery.nnz),
        gallery_feature_seconds=time.perf_counter() - start,
        gallery_cache_sha256=hash_file(out / "cache/gallery/csr.json"),
        artifact_hashes={
            name: hash_file(dest / name)
            for name in ("model.pt", "gallery_ids.npy", "gallery_uids.json")
        },
        research_note="Fit and gallery use Session1 only; no Session2 ROI read.",
        approximation=bool(args.max_patches_per_image),
        classifier_plan={
            "selection": "Session1 top variance 1024 columns; L2",
            "linear_svm_C": 1.0,
            "crc_lambda": 0.1,
            "reduced_1nn_control": True,
        },
    )
    atomic_json(dest / "freeze.json", frozen)
    LOG.info("M5 frozen: %d dimensions, %d gallery nonzeros", gallery.shape[1], gallery.nnz)
    print(
        json.dumps(
            {
                k: frozen[k]
                for k in ("status", "feature_dim", "fit_seconds", "gallery_feature_seconds")
            },
            indent=2,
        ),
        flush=True,
    )


def _gallery(dest, out, frozen):
    cache = out / "cache/gallery"
    if hash_file(cache / "csr.json") != frozen["gallery_cache_sha256"]:
        raise RuntimeError("Gallery cache metadata changed after freezing.")
    meta = json.loads((cache / "csr.json").read_text())
    for name, digest in meta["hashes"].items():
        if hash_file(cache / name) != digest:
            raise RuntimeError(f"Gallery cache corrupted: {name}")
    return load_csr(cache), np.load(dest / "gallery_ids.npy", allow_pickle=False)


def save_result(out, name, scores, classes, probes, extra):
    ranked, ranked_scores = ranking(scores, classes)
    labels = probes.palm_id.to_numpy(dtype=np.int32)
    metric = compute_metrics(labels, ranked, classes)
    metric.update(extra)
    metric.update(
        model=name, variant=VARIANT, rank1_ci95_palm_bootstrap=bootstrap_rank1(labels, ranked[:, 0])
    )
    predictions = pd.DataFrame(
        {
            "sample_uid": probes.sample_uid,
            "true_palm_id": labels,
            "pred_rank1": ranked[:, 0],
            "correct_rank1": (ranked[:, 0] == labels).astype(int),
        }
    )
    for k in range(min(5, len(classes))):
        predictions[f"pred_rank{k + 1}"] = ranked[:, k]
        predictions[f"score_rank{k + 1}"] = ranked_scores[:, k]
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(dest / "predictions.csv", index=False)
    np.savez_compressed(dest / "scores.npz", scores=scores, classes=classes, ranked=ranked)
    pd.DataFrame(
        {
            "rank": np.arange(1, len(classes) + 1),
            "accuracy": [
                (ranked[:, :k] == labels[:, None]).any(axis=1).mean()
                for k in range(1, len(classes) + 1)
            ],
        }
    ).to_csv(dest / "cmc.csv", index=False)
    metric["prediction_sha256"] = hash_file(dest / "predictions.csv")
    metric["scores_sha256"] = hash_file(dest / "scores.npz")
    atomic_json(dest / "metrics.json", metric)
    print(json.dumps(metric, indent=2, ensure_ascii=False), flush=True)
    return metric


def evaluate(args):
    ws, dest, out, vendor = _paths(args)
    if (out / "nearest_template/metrics.json").exists():
        raise FileExistsError("M5 Session2 results exist; refuse overwrite.")
    frame, root, manifest_sha = read_context(ws, args.data_root)
    _, PalmPCAGabor, info = get_vendor(vendor)
    frozen = _check_frozen(dest, manifest_sha, info)
    gallery, ids = _gallery(dest, out, frozen)
    probes, paths = paths_for(frame, root, "session2", verify_hash=not args.skip_sha)
    model = PalmPCAGabor.load(dest / "model.pt", args.device)
    start = time.perf_counter()
    X = extract_cached(
        model,
        paths,
        probes,
        out / "cache/probe",
        frozen["model_sha256"],
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    query_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if args.matcher == "hybrid":
        from palmprint.models.palmnet_matching import hybrid_nearest_class_scores

        classes, scores = hybrid_nearest_class_scores(
            X,
            gallery,
            ids,
            batch_size=args.match_batch,
            gallery_batch_size=args.gallery_batch,
            threads=args.threads,
        )
    else:
        classes, scores = nearest_class_scores(
            X, gallery, ids, batch_size=args.match_batch, gallery_batch_size=args.gallery_batch
        )
    elapsed = time.perf_counter() - start
    save_result(
        out,
        "nearest_template",
        scores,
        classes,
        probes,
        {
            "evaluation": "Tongji Session1 gallery / Session2 probe; 600 closed-set Palm IDs",
            "score_kind": "min squared Euclidean over 10 gallery templates per Palm ID",
            "manifest_sha256": manifest_sha,
            "model_sha256": frozen["model_sha256"],
            "vendor": info,
            "feature_dim": frozen["feature_dim"],
            "matching_backend": args.matcher,
            "matching_threads": args.threads,
            "adapter_source_sha256": hash_file(__file__),
            "hash_verification": frozen["hash_verification"] and not args.skip_sha,
            "fit_seconds": frozen["fit_seconds"],
            "gallery_feature_seconds": frozen["gallery_feature_seconds"],
            "probe_feature_seconds": query_seconds,
            "matching_seconds": elapsed,
            "matching_ms_per_probe_batch_amortized": 1000 * elapsed / len(probes),
            "probe_feature_ms_per_probe_batch_amortized": 1000 * query_seconds / len(probes),
            "timing_note": "Batch amortized, includes sparse cache IO; resume may reuse cache. "
            "Not comparable to existing single-query CPU latency.",
            "approximation": frozen["approximation"],
        },
    )


def self_test():
    gallery = sparse.csr_matrix(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0], [0.0, 0.8, 0.2]]
    )
    ids = np.array([1, 2, 1, 2])
    probes = sparse.csr_matrix([[0.85, 0.15, 0.0], [0.0, 0.85, 0.15], [1.0, 0.0, 0.0]])
    classes, scores = nearest_class_scores(probes, gallery, ids, batch_size=2, gallery_batch_size=1)
    assert classes[np.argmin(scores, axis=1)].tolist() == [1, 2, 1]
    print("SELF-TEST PASSED: sparse Euclidean + classwise template minima.")


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default=".")
    p.add_argument("--palmnet-python", help="Default .runtime/PalmNet/python")
    p.add_argument("--data-root", help="Default workspace/data/raw; overrides local_dataset.json")
    p.add_argument("--skip-sha", action="store_true", help="Non-audited speed check only")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    for name in ("train-classifiers", "evaluate-classifiers"):
        c = commands.add_parser(name)
        c.add_argument("--threads", type=_positive, default=4)
    for name in ("train", "evaluate"):
        c = commands.add_parser(name)
        c.add_argument("--device", default="cpu")
        c.add_argument("--threads", type=_positive, default=4)
        c.add_argument("--batch-size", type=_positive, default=2)
        c.add_argument("--chunk-size", type=_positive, default=64)
        if name == "train":
            c.add_argument("--workers", type=int, default=0)
            c.add_argument("--seed", type=int, default=42)
            c.add_argument("--pca-filters", type=_positive, default=15)
            c.add_argument("--fixed-orientations", type=_positive, default=10)
            c.add_argument("--adaptive-orientations", type=int, default=10)
            c.add_argument("--selected-wavelets", type=int, default=5)
            c.add_argument("--hist-block", type=_positive, default=23)
            c.add_argument("--dtype", choices=["float64", "float32"], default="float64")
            c.add_argument(
                "--backend", choices=["auto", "reference", "optimized"], default="optimized"
            )
            c.add_argument("--convolution", choices=["auto", "direct", "fft"], default="auto")
            c.add_argument("--response-batch", type=_positive, default=16)
            c.add_argument("--pca-image-batch", type=_positive, default=4)
            c.add_argument("--orientation-batch", type=_positive, default=16)
            c.add_argument("--gabor-tuning-batch", type=_positive, default=2)
            c.add_argument("--max-patches-per-image", type=_positive, default=None)
        else:
            c.add_argument("--match-batch", type=_positive, default=64)
            c.add_argument("--gallery-batch", type=_positive, default=256)
            c.add_argument("--matcher", choices=["scipy", "hybrid"], default="hybrid")
    return p


def main(argv=None):
    args = cli().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "self-test":
        return self_test()
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    torch.sparse.check_sparse_tensor_invariants.enable()
    if args.command in ("train-classifiers", "evaluate-classifiers"):
        from palmprint.training.palmnet_classifiers import evaluate_classifiers, train_classifiers

        return {
            "train-classifiers": train_classifiers,
            "evaluate-classifiers": evaluate_classifiers,
        }[args.command](args)
    return {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
