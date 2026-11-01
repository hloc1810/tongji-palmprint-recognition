"""Frozen ResNet18 512-D cosine-gallery identification.

Does not train any new parameters or alter the original 600-logit baseline.
Gallery: session1 only; query/test: session2 only. Both use eval transforms.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..data.dataset import read_gray
from ..data.preparation import load_manifest
from ..models.cnn_runtime import inference_context, make_loader, resolve_device
from ..models.resnet import make_resnet, preprocessing, torch_modules
from ..reporting.evidence import write_prediction_outputs
from ..utils import atomic_json, hash_file
from .metrics import bootstrap_rank1, compute_metrics, matrix, ranking

STRATEGIES = ("nearest", "centroid")
EMBED_DIM = 512


def normalize_rows(array: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Unit-normalize 2-D features, rejecting degenerate/nonfinite features."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != EMBED_DIM:
        raise ValueError(f"Expected (N,{EMBED_DIM}) feature matrix")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings include nonfinite values")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.all(norms > eps):
        raise ValueError("Encountered zero/near-zero embedding")
    return np.ascontiguousarray((values / norms[:, None]).astype(np.float32))


def class_cosine_scores(
    queries: np.ndarray,
    gallery: np.ndarray,
    gallery_ids: np.ndarray,
    classes: np.ndarray,
    strategy: str = "nearest",
    chunk_size: int = 128,
) -> np.ndarray:
    """Return (num_queries,num_classes) negative cosine scores (lower is better).

    nearest: max cosine over *all* images of each class, then negate.
    centroid: mean of normalized class features, renormalized before cosine.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose {STRATEGIES}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    queries = normalize_rows(queries)
    gallery = normalize_rows(gallery)
    gallery_ids = np.asarray(gallery_ids, dtype=np.int32)
    classes = np.asarray(classes, dtype=np.int32)
    if len(gallery) != len(gallery_ids) or len(classes) == 0:
        raise ValueError("Gallery size/classes mismatch")
    if not np.array_equal(classes, np.sort(np.unique(classes))):
        raise ValueError("Classes must be unique and sorted")
    if not np.array_equal(np.unique(gallery_ids), classes):
        raise ValueError("Every class must have at least one gallery embedding")
    # Sorting the gallery once makes classwise reduceat efficient and deterministic.
    if strategy == "nearest":
        order = np.argsort(gallery_ids, kind="stable")
        gallery_sorted = gallery[order]
        sorted_ids = gallery_ids[order]
        first = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        output = np.empty((len(queries), len(classes)), dtype=np.float32)
        for start in range(0, len(queries), chunk_size):
            sim = queries[start : start + chunk_size] @ gallery_sorted.T
            output[start : start + chunk_size] = -np.maximum.reduceat(sim, first, axis=1)
    else:
        centroids = np.stack([gallery[gallery_ids == c].mean(axis=0) for c in classes])
        centroids = normalize_rows(centroids)
        output = np.empty((len(queries), len(classes)), dtype=np.float32)
        for start in range(0, len(queries), chunk_size):
            output[start : start + chunk_size] = -(
                queries[start : start + chunk_size] @ centroids.T
            )
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite cosine scores")
    return output


def _load_frozen_network(ws: Path, seed: int, device: str, *, dev_checkpoint: bool = False):
    """Strict checkpoint/manifest binding; dev checkpoint is never the final model."""
    torch, nn, *_ = torch_modules()
    df, info = load_manifest(ws)
    artifact = ws / "artifacts" / f"resnet18_seed{seed}.pth"
    freeze_path = ws / "artifacts" / f"resnet18_seed{seed}_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest_path = ws / "data/metadata/manifest.csv"
    if (
        hash_file(artifact) != frozen["artifact_sha256"]
        or hash_file(manifest_path) != frozen["manifest_sha256"]
    ):
        raise RuntimeError("Frozen ResNet18 artifact or manifest has changed")
    ckpt = torch.load(artifact, map_location="cpu", weights_only=True)
    classes = np.arange(1, int(ckpt["num_classes"]) + 1, dtype=np.int32)
    if not np.array_equal(np.sort(df[df.session == "session1"].palm_id.unique()), classes):
        raise RuntimeError("Checkpoint class mapping disagrees with session1 manifest")
    dev_path = ws / "outputs/runs" / f"resnet18_seed{seed}" / "best_dev.pth"
    if dev_checkpoint:
        if not dev_path.exists():
            raise FileNotFoundError("best_dev.pth required for session1-only validation")
        state_dict = torch.load(dev_path, map_location="cpu", weights_only=True)["state_dict"]
        checkpoint_hash = hash_file(dev_path)
    else:
        state_dict = ckpt["state_dict"]
        checkpoint_hash = frozen["artifact_sha256"]
    net = make_resnet(int(ckpt["num_classes"]), "none")
    net.load_state_dict(state_dict, strict=True)
    if net.fc.in_features != EMBED_DIM:
        raise ValueError("ResNet18 penultimate feature dimension must be 512")
    net.fc = nn.Identity()  # after avgpool/flatten; pre-FC 512-D features
    actual_device = resolve_device(device)
    net = net.to(actual_device).eval()
    return net, actual_device, df, info, classes, int(ckpt["image_size"]), checkpoint_hash, frozen


def _extract(model, frame, root: Path, batch_size: int, image_size: int, device):
    torch, *_ = torch_modules()
    if frame.empty:
        raise ValueError("Cannot extract features from an empty session")
    loader = make_loader(frame, root, batch_size, False, image_size)
    out = []
    model.eval()
    with inference_context(torch, device):
        for images, _ in loader:
            features = model(images.to(device)).detach().to("cpu", dtype=torch.float32)
            if features.ndim != 2 or features.shape[1] != EMBED_DIM:
                raise ValueError(f"Expected Bx512 output, got {tuple(features.shape)}")
            out.append(features.numpy())
    result = normalize_rows(np.concatenate(out, axis=0))
    if len(result) != len(frame):
        raise RuntimeError("Feature order differs from manifest order")
    return result


def _check_session_split(gallery_df, query_df, classes):
    if (
        gallery_df.sample_uid.duplicated().any()
        or query_df.sample_uid.duplicated().any()
        or set(gallery_df.sample_uid) & set(query_df.sample_uid)
    ):
        raise RuntimeError("Gallery and query overlap or contain duplicate samples")
    if not np.array_equal(np.sort(gallery_df.palm_id.unique()), classes):
        raise RuntimeError("Gallery must contain every closed-set Palm ID")
    if not np.isin(query_df.palm_id.to_numpy(), classes).all():
        raise RuntimeError("Query IDs are outside gallery closed-set classes")


def _load_or_build_gallery(
    ws: Path,
    seed: int,
    *,
    model,
    device,
    frame,
    root: Path,
    image_size: int,
    checkpoint_hash: str,
    manifest_sha256: str,
    batch_size: int,
):
    """Cache features from session1 only, bound to both model and manifest hashes."""
    file = ws / "artifacts" / f"resnet18_seed{seed}_gallery512.npz"
    expected_uid = np.asarray(frame.sample_uid.tolist(), dtype=str)
    expected_ids = frame.palm_id.to_numpy(dtype=np.int32)
    if file.exists():
        with np.load(file, allow_pickle=False) as data:
            if (
                str(data["checkpoint_sha256"].item()) != checkpoint_hash
                or str(data["manifest_sha256"].item()) != manifest_sha256
                or int(data["image_size"].item()) != image_size
                or not np.array_equal(data["sample_uid"], expected_uid)
                or not np.array_equal(data["palm_id"], expected_ids)
            ):
                raise RuntimeError("Gallery cache does not match frozen checkpoint/manifest")
            features = normalize_rows(data["features"])
        if features.shape[0] != len(frame):
            raise RuntimeError("Cached Gallery has incorrect size")
        return features, expected_ids, file, False
    features = _extract(model, frame, root, batch_size, image_size, device)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file,
        features=features,
        palm_id=expected_ids,
        sample_uid=expected_uid,
        checkpoint_sha256=np.asarray(checkpoint_hash),
        manifest_sha256=np.asarray(manifest_sha256),
        image_size=np.asarray(image_size),
    )
    return features, expected_ids, file, True


def build_gallery(ws: Path, *, seed: int = 17, batch_size: int = 32, device: str = "auto") -> dict:
    """Create/check the Session1-only frozen gallery without touching Session2 images."""
    ws = Path(ws)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    net, dev, df, info, classes, size, checkpoint_hash, freeze = _load_frozen_network(
        ws, seed, device
    )
    gallery_df = df[df.session == "session1"].reset_index(drop=True)
    if not np.array_equal(np.sort(gallery_df.palm_id.unique()), classes):
        raise RuntimeError("Gallery lacks some frozen CNN identities")
    features, ids, gallery_file, created = _load_or_build_gallery(
        ws,
        seed,
        model=net,
        device=dev,
        frame=gallery_df,
        root=Path(info["data_root"]),
        image_size=size,
        checkpoint_hash=checkpoint_hash,
        manifest_sha256=freeze["manifest_sha256"],
        batch_size=batch_size,
    )
    return {
        "seed": seed,
        "n_gallery": len(ids),
        "classes": len(classes),
        "dimension": features.shape[1],
        "created": created,
        "gallery_file": str(gallery_file),
        "sha256": hash_file(gallery_file),
        "source": "session1 only",
        "device": str(dev),
    }


def evaluate_embeddings(
    ws: Path,
    *,
    seed: int = 17,
    batch_size: int = 32,
    device: str = "auto",
    strategies: Sequence[str] = STRATEGIES,
    chunk_size: int = 128,
) -> dict:
    """Run frozen cross-session evaluation; never refit parameters or tune on session2."""
    ws = Path(ws)
    strategies = tuple(dict.fromkeys(strategies))
    if not strategies or any(s not in STRATEGIES for s in strategies):
        raise ValueError(f"Strategies must be drawn from {STRATEGIES}")
    if batch_size < 1 or chunk_size < 1:
        raise ValueError("batch_size/chunk_size must be positive")
    outdirs = {s: ws / "outputs/runs" / f"resnet18_seed{seed}_embedding_{s}" for s in strategies}
    for dst in outdirs.values():
        if (dst / "metrics.json").exists():
            raise RuntimeError(f"Final test already evaluated: {dst}")
    net, dev, df, info, classes, size, checkpoint_hash, freeze = _load_frozen_network(
        ws, seed, device
    )
    gallery_df = df[df.session == "session1"].reset_index(drop=True)
    test_df = df[df.session == "session2"].reset_index(drop=True)
    _check_session_split(gallery_df, test_df, classes)
    root = Path(info["data_root"])
    start = time.perf_counter()
    gallery, ids, cache, created = _load_or_build_gallery(
        ws,
        seed,
        model=net,
        device=dev,
        frame=gallery_df,
        root=root,
        image_size=size,
        checkpoint_hash=checkpoint_hash,
        manifest_sha256=freeze["manifest_sha256"],
        batch_size=batch_size,
    )
    gallery_secs = time.perf_counter() - start
    start = time.perf_counter()
    queries = _extract(net, test_df, root, batch_size, size, dev)
    query_secs = time.perf_counter() - start
    result = {}
    for strategy, outdir in outdirs.items():
        t = time.perf_counter()
        score = class_cosine_scores(queries, gallery, ids, classes, strategy, chunk_size)
        compute_secs = time.perf_counter() - t
        ranked, ranked_scores = ranking(score, classes)
        metrics = compute_metrics(test_df.palm_id.to_numpy(dtype=np.int32), ranked, classes)
        metrics.update(
            {
                "model": "resnet18_embedding",
                "seed": seed,
                "method": f"ResNet18 512-D {strategy} cosine",
                "score_kind": "negative_cosine_similarity",
                "gallery_strategy": strategy,
                "model_definition": {
                    "method": f"ResNet18 ROI 512-D {strategy} cosine",
                    "score_kind": "negative_cosine_similarity",
                },
                "embedding_dimension": EMBED_DIM,
                "l2_normalized": True,
                "n_gallery": int(len(gallery)),
                "n_probes": int(len(queries)),
                "manifest_sha256": freeze["manifest_sha256"],
                "artifact_sha256": checkpoint_hash,
                "gallery_sha256": hash_file(cache),
                "evaluation": "cross-session",
                "gallery_session": "session1",
                "probe_session": "session2",
                "device": str(dev),
                "gallery_cache_created": created,
                "gallery_wall_ms": 1000 * gallery_secs,
                "probe_feature_ms_per_image_batch_amortized": 1000 * query_secs / len(queries),
                "cosine_matching_ms_per_image_batch_amortized": 1000 * compute_secs / len(queries),
                "rank1_ci95_palm_bootstrap": bootstrap_rank1(
                    test_df.palm_id.to_numpy(dtype=np.int32), ranked[:, 0]
                ),
                "warning": "Exploratory post-hoc comparison after prior Session2 benchmark results. "
                "Do not select a preferred strategy using Session2 results.",
            }
        )
        outdir.mkdir(parents=True, exist_ok=True)
        write_prediction_outputs(
            outdir, test_df, ranked, ranked_scores, metrics, classes, unsorted_scores=score
        )
        np.save(
            outdir / "confusion_matrix.npy",
            matrix(test_df.palm_id.to_numpy(dtype=np.int32), ranked, classes),
        )
        atomic_json(outdir / "metrics.json", metrics)
        result[strategy] = metrics
    return result


def validate_embeddings(
    ws: Path,
    *,
    seed: int = 17,
    batch_size: int = 32,
    device: str = "auto",
    strategies: Sequence[str] = STRATEGIES,
    chunk_size: int = 128,
) -> dict:
    """Within-session check with best_dev checkpoint, gallery folds 1-4, query fold 0.

    Note: fold0 also selected checkpoint epoch; this is diagnostic, not independent CV.
    """
    ws = Path(ws)
    net, dev, df, info, classes, size, chk_hash, _ = _load_frozen_network(
        ws, seed, device, dev_checkpoint=True
    )
    session1 = df[df.session == "session1"]
    gallery_df = session1[session1.fold != 0].reset_index(drop=True)
    query_df = session1[session1.fold == 0].reset_index(drop=True)
    _check_session_split(gallery_df, query_df, classes)
    root = Path(info["data_root"])
    gallery = _extract(net, gallery_df, root, batch_size, size, dev)
    queries = _extract(net, query_df, root, batch_size, size, dev)
    run_dir = ws / "outputs/runs" / f"resnet18_seed{seed}"
    out = {}
    for s in tuple(dict.fromkeys(strategies)):
        sc = class_cosine_scores(
            queries, gallery, gallery_df.palm_id.to_numpy(), classes, s, chunk_size
        )
        rank, _ = ranking(sc, classes)
        m = compute_metrics(query_df.palm_id.to_numpy(), rank, classes)
        m.update(
            {
                "seed": seed,
                "strategy": s,
                "checkpoint_sha256": chk_hash,
                "gallery": "session1 fold!=0",
                "query": "session1 fold==0",
                "warning": "Diagnostic, not independent validation: fold0 was used for "
                "checkpoint selection during original training.",
            }
        )
        atomic_json(run_dir / f"embedding_dev_{s}.json", m)
        out[s] = m
    return out


def predict_one_embedding(
    ws: Path, seed: int, image_path: Path, *, strategy: str = "nearest", device: str = "cpu"
) -> dict:
    """Identify one official 128x128 ROI against frozen session1 gallery."""
    ws = Path(ws)
    net, dev, df, info, classes, size, checkpoint_hash, freeze = _load_frozen_network(
        ws, seed, device
    )
    gallery_df = df[df.session == "session1"].reset_index(drop=True)
    cache = ws / "artifacts" / f"resnet18_seed{seed}_gallery512.npz"
    if not cache.exists():
        raise FileNotFoundError("Gallery not built. Run embedding-gallery first.")
    with np.load(cache, allow_pickle=False) as data:
        if (
            str(data["checkpoint_sha256"].item()) != checkpoint_hash
            or str(data["manifest_sha256"].item()) != freeze["manifest_sha256"]
            or int(data["image_size"].item()) != size
            or not np.array_equal(
                data["sample_uid"], np.asarray(gallery_df.sample_uid.tolist(), dtype=str)
            )
            or not np.array_equal(data["palm_id"], gallery_df.palm_id.to_numpy())
        ):
            raise RuntimeError("Stored Gallery does not match frozen training artifacts")
        gallery = normalize_rows(data["features"])
        ids = data["palm_id"].astype(np.int32)
    torch, *_ = torch_modules()
    before = time.perf_counter()
    tensor = preprocessing(False, size)(Image.fromarray(read_gray(image_path)))[None]
    net.eval()
    with inference_context(torch, dev):
        feature = net(tensor.to(dev)).detach().to("cpu", dtype=torch.float32).numpy()
    score = class_cosine_scores(feature, gallery, ids, classes, strategy)
    ordered, ordered_scores = ranking(score, classes)
    return {
        "model": "resnet18_embedding",
        "seed": seed,
        "strategy": strategy,
        "predicted_palm_id": int(ordered[0, 0]),
        "top5": ordered[0, :5].tolist(),
        "scores_lower_is_better": ordered_scores[0, :5].astype(float).tolist(),
        "cosine_top5": (-ordered_scores[0, :5]).astype(float).tolist(),
        "elapsed_ms": 1000 * (time.perf_counter() - before),
        "score_kind": "negative_cosine_similarity",
    }
