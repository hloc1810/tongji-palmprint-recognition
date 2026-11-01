"""Frozen ResNet18 evaluation and one-sample inference."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.dataset import read_gray
from ..data.preparation import load_manifest
from ..models.resnet import make_resnet, preprocessing, torch_modules
from ..reporting.evidence import write_prediction_outputs
from ..training.cnn import _loader, predict_loader
from ..utils import atomic_json, hash_file
from .metrics import bootstrap_rank1, compute_metrics, matrix, ranking


def evaluate_one(ws: Path, *, seed: int = 17, batch_size: int = 32):
    torch, *_ = torch_modules()
    df, info = load_manifest(ws)
    path = ws / "artifacts" / f"resnet18_seed{seed}.pth"
    frozen = json.loads((ws / "artifacts" / f"resnet18_seed{seed}_freeze.json").read_text())
    out = ws / "outputs/runs" / f"resnet18_seed{seed}"
    if (out / "metrics.json").exists():
        raise RuntimeError("Final test already evaluated for this CNN seed")
    if hash_file(path) != frozen["artifact_sha256"] or hash_file(
            ws / "data/metadata/manifest.csv") != frozen["manifest_sha256"]:
        raise RuntimeError("Frozen CNN or manifest was modified")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = make_resnet(ckpt["num_classes"], "none")
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_df = df[df.session == "session2"].reset_index(drop=True)
    loader = _loader(test_df, Path(info["data_root"]), batch_size, False,
                     ckpt["image_size"])
    start = time.perf_counter()
    logits, labels = predict_loader(model, loader, device)
    elapsed = time.perf_counter() - start
    ordered, scores = ranking(-logits, np.arange(1, ckpt["num_classes"]+1))
    m = compute_metrics(labels, ordered, np.arange(1, ckpt["num_classes"]+1))
    m.update({"model": "resnet18", "seed": seed,
              "manifest_sha256": frozen["manifest_sha256"],
              "artifact_sha256": frozen["artifact_sha256"],
              "score_kind": "negative_logit",
              "model_definition": {"method": "ResNet18 ROI classification",
                                   "score_kind": "negative_logit"},
              "rank1_ci95_palm_bootstrap": bootstrap_rank1(labels, ordered[:, 0]),
              "matching_ms_per_probe_batch_amortized": 1000*elapsed/len(labels),
              "evaluation": "cross-session", "device": str(device),
              "warning": "Historic pilot used part of session2; test not fully untouched."})
    out.mkdir(parents=True, exist_ok=True)
    write_prediction_outputs(out, test_df, ordered, scores, m,
                             np.arange(1, ckpt["num_classes"]+1), unsorted_scores=-logits)
    np.save(out / "confusion_matrix.npy", matrix(labels, ordered,
                                                   np.arange(1, ckpt["num_classes"]+1)))
    atomic_json(out / "metrics.json", m)
    return m



def predict_one_cnn(ws: Path, seed: int, image_path: Path) -> dict:
    """Single ROI prediction for the Streamlit demo; do not infer label from filename."""
    torch, *_ = torch_modules()
    file_path = ws / "artifacts" / f"resnet18_seed{seed}.pth"
    freeze_path = ws / "artifacts" / f"resnet18_seed{seed}_freeze.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if hash_file(file_path) != frozen["artifact_sha256"]:
        raise RuntimeError("CNN demo artifact differs from frozen weights")
    checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
    network = make_resnet(checkpoint["num_classes"], "none")
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()
    start = time.perf_counter()
    tensor = preprocessing(False, checkpoint["image_size"])(
        Image.fromarray(read_gray(image_path)))[None]
    with torch.inference_mode():
        logits = network(tensor).cpu().numpy()
    ordered, values = ranking(-logits, np.arange(1, checkpoint["num_classes"]+1))
    return {"model": "resnet18", "seed": seed,
            "predicted_palm_id": int(ordered[0, 0]),
            "top5": ordered[0, :5].tolist(),
            "scores_lower_is_better": values[0, :5].tolist(),
            "elapsed_ms": 1000*(time.perf_counter() - start),
            "score_kind": "negative_logit"}
