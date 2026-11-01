"""Embedding matching tests use synthetic data only, no real Tongji scans."""

import numpy as np
import pytest

from palmprint.evaluation.embedding import EMBED_DIM, class_cosine_scores, normalize_rows
from palmprint.evaluation.metrics import compute_metrics, ranking


def feature(x, y):
    z = np.zeros(EMBED_DIM, dtype=np.float32)
    z[0] = x
    z[1] = y
    return z


def test_exact_nearest_and_centroid_scoring():
    g = np.stack([feature(1, 0), feature(0.8, 0.6), feature(0, 1), feature(0, 0.9)])
    ids = np.array([1, 1, 2, 2])
    q = np.stack([feature(1, 0), feature(0, 1)])
    for method in ("nearest", "centroid"):
        scores = class_cosine_scores(q, g, ids, np.array([1, 2]), method, chunk_size=1)
        ranked, _ = ranking(scores, np.array([1, 2]))
        assert ranked[:, 0].tolist() == [1, 2]
        assert compute_metrics(np.array([1, 2]), ranked, np.array([1, 2]))["rank1"] == 1.0
        assert scores.shape == (2, 2)
        assert (scores >= -1.000001).all() and (scores <= 1.000001).all()


def test_reject_invalid_gallery_and_features():
    g = np.stack([feature(1, 0)])
    with pytest.raises(ValueError, match="zero"):
        normalize_rows(np.zeros((1, EMBED_DIM)))
    with pytest.raises(ValueError, match="Every class"):
        class_cosine_scores(g, g, np.array([1]), np.array([1, 2]))
    with pytest.raises(ValueError, match="Unknown strategy"):
        class_cosine_scores(g, g, np.array([1]), np.array([1]), "wrong")


def test_frozen_checkpoint_gallery_evaluate_and_predict(tmp_path):
    torch = pytest.importorskip("torch")
    from palmprint.data.dataset import generate_synthetic_roi
    from palmprint.data.preparation import prepare
    from palmprint.evaluation.embedding import (
        build_gallery,
        evaluate_embeddings,
        predict_one_embedding,
    )
    from palmprint.models.resnet import make_resnet
    from palmprint.utils import atomic_json, hash_file

    # Synthetic only: small three-ID miniature that verifies end-to-end I/O.
    root = tmp_path / "synthetic"
    ws = tmp_path / "ws"
    generate_synthetic_roi(root, palms=3, samples=2)
    prepare(ws, root, strict=False, folds=2)
    model = make_resnet(3, "none")
    ckpt_path = ws / "artifacts" / "resnet18_seed17.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "num_classes": 3, "image_size": 32, "seed": 17},
        ckpt_path,
    )
    atomic_json(
        ws / "artifacts" / "resnet18_seed17_freeze.json",
        {
            "artifact_sha256": hash_file(ckpt_path),
            "manifest_sha256": hash_file(ws / "data/metadata/manifest.csv"),
        },
    )
    built = build_gallery(ws, seed=17, batch_size=6, device="cpu")
    assert built["created"] is True
    assert built["n_gallery"] == 6
    assert build_gallery(ws, seed=17, batch_size=6, device="cpu")["created"] is False
    result = evaluate_embeddings(
        ws, seed=17, batch_size=6, device="cpu", strategies=("nearest", "centroid"), chunk_size=2
    )
    assert result["nearest"]["n"] == 6
    assert result["centroid"]["classes"] == 3
    assert result["nearest"]["embedding_dimension"] == 512
    gallery_path = ws / "artifacts/resnet18_seed17_gallery512.npz"
    with np.load(gallery_path, allow_pickle=False) as data:
        assert data["features"].shape == (6, 512)
        assert np.allclose(np.linalg.norm(data["features"], axis=1), 1, atol=1e-5)
        assert data["sample_uid"].tolist()[0].startswith("session1/")
    from palmprint.reporting.evidence import verify_run_evidence

    for strategy in ("nearest", "centroid"):
        run_dir = ws / "outputs/runs" / f"resnet18_seed17_embedding_{strategy}"
        verified = verify_run_evidence(run_dir)
        assert verified is not None
    from palmprint.reporting.summary import generate_report

    full_report = generate_report(ws)
    assert set(full_report.model.tolist()) == {
        "resnet18_seed17_embedding_nearest",
        "resnet18_seed17_embedding_centroid",
    }
    sample = root / "session2" / "00001.png"
    p = predict_one_embedding(ws, 17, sample, strategy="nearest", device="cpu")
    assert p["predicted_palm_id"] in (1, 2, 3)
    assert len(p["top5"]) == 3
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_embeddings(ws, seed=17, strategies=("nearest",), device="cpu")
