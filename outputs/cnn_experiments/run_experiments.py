"""Registered three-seed ResNet18 experiment; run from the repository root."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import torch
import torch_directml
from PIL import Image

from palmprint.data.dataset import read_gray, resolve_path, validate_manifest
from palmprint.data.preparation import load_manifest
from palmprint.evaluation.cnn import evaluate_one
from palmprint.evaluation.metrics import bootstrap_delta
from palmprint.models.resnet import make_resnet, preprocessing
from palmprint.reporting.cnn_comparison import paired_seed_mean_delta
from palmprint.reporting.evidence import verify_run_evidence
from palmprint.reporting.summary import generate_report
from palmprint.training.cnn import resolve_device, train_one
from palmprint.training.selection import _classical_source_hash
from palmprint.utils import atomic_json, env_info, hash_file, hash_object, read_yaml

SEEDS = (17, 29, 43)
DEVICE = "directml:0"


def now():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def cnn_source_hash(ws):
    paths = sorted((ws / "src/palmprint").rglob("*.py"))
    return hash_object({str(p.relative_to(ws)): hash_file(p) for p in paths})


def cpu_latency(ws, seed, test, root):
    """Same read+preprocess+score timing boundary as the classical CPU benchmark."""
    torch.set_num_threads(1)
    device = torch.device("cpu")
    checkpoint = torch.load(ws / f"artifacts/resnet18_seed{seed}.pth", map_location="cpu", weights_only=True)
    net = make_resnet(checkpoint["num_classes"], "none").eval()
    net.load_state_dict(checkpoint["state_dict"])
    transform = preprocessing(False, checkpoint["image_size"])
    warmup = transform(Image.fromarray(read_gray(resolve_path(root, test.path.iloc[0]))))[None]
    with torch.inference_mode():
        for _ in range(3):
            net(warmup)
        timings = []
        for index, row in enumerate(test.itertuples(index=False)):
            started = time.perf_counter()
            gray = read_gray(resolve_path(root, row.path))
            read_end = time.perf_counter()
            tensor = transform(Image.fromarray(gray))[None]
            prep_end = time.perf_counter()
            scores = -net(tensor).numpy()
            end = time.perf_counter()
            if scores.shape != (1, 600) or not np.isfinite(scores).all():
                raise RuntimeError("Invalid CPU benchmark output")
            timings.append({"sample_uid": row.sample_uid, "read_ms": (read_end-started)*1000,
                            "transform_ms": (prep_end-read_end)*1000,
                            "network_ms": (end-prep_end)*1000, "total_ms": (end-started)*1000})
            if (index + 1) % 1000 == 0:
                print(f"CPU latency seed={seed}: {index+1}/6000", flush=True)
    table = pd.DataFrame(timings)
    run = ws / f"outputs/runs/resnet18_seed{seed}"
    table.to_csv(run / "cpu_latency_per_probe.csv", index=False)
    result = {"n_queries": len(table), "device": str(device), "threads": 1,
              "warmup_calls": 3, "repetitions": 1, "model_loading_included": False,
              "timing": "read+decode ROI, tensor transforms, network logits; excludes ranking/UI",
              "total_median_ms": float(table.total_ms.median()),
              "total_p95_ms": float(table.total_ms.quantile(.95)),
              "network_median_ms": float(table.network_ms.median())}
    atomic_json(run / "benchmark_cpu.json", result)
    return result


def main():
    ws = Path.cwd()
    out = ws / "outputs/cnn_experiments"
    out.mkdir(parents=True, exist_ok=True)
    # Register DirectML before any autograd backward in this process.
    resolve_device(DEVICE)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    config_path = ws / "configs/resnet18.yaml"
    conf = read_yaml(config_path)
    assert tuple(conf["seeds"]) == SEEDS and conf["weights"] == "imagenet1k_v1"
    manifest, dataset = load_manifest(ws)
    validate_manifest(manifest, folds=5, strict=True)
    test = manifest[manifest.session == "session2"].reset_index(drop=True)
    root = Path(dataset["data_root"])
    classical_lock = json.loads((ws / "artifacts/protocol_lock.json").read_text(encoding="utf-8"))
    assert classical_lock["source_sha256"] == _classical_source_hash(ws)
    config = {k: v for k, v in conf.items() if k not in ("seeds", "image_size")}
    config.update(size=conf["image_size"], device=DEVICE)
    state_path = out / "progress.json"
    events = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else []

    def stage(name, action):
        event = {"stage": name, "status": "running", "started_at": now()}
        events.append(event)
        atomic_json(state_path, events)
        print(f"[{event['started_at']}] START {name}", flush=True)
        started = time.perf_counter()
        try:
            value = action()
        except Exception as exc:
            event.update(status="failed", error=repr(exc), finished_at=now(),
                         wall_seconds=time.perf_counter()-started)
            atomic_json(state_path, events)
            raise
        event.update(status="completed", finished_at=now(), wall_seconds=time.perf_counter()-started)
        atomic_json(state_path, events)
        print(f"[{event['finished_at']}] DONE {name}: {event['wall_seconds']:.3f}s", flush=True)
        if isinstance(value, dict):
            print(json.dumps(value, ensure_ascii=False, default=str), flush=True)
        return value

    registered = out / "protocol_before_cnn_test.json"
    if not registered.exists():
        if any((ws / f"artifacts/resnet18_seed{s}_freeze.json").exists() for s in SEEDS):
            raise RuntimeError("CNN freezes predate registration")
        weights = ws / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        classical_evidence = {}
        for name in ("knn", "svm", "crc"):
            stage(f"verify_existing_{name}", lambda name=name: verify_run_evidence(
                ws / f"outputs/runs/{name}", expected_probes=test,
                manifest_sha256=dataset["manifest_sha256"]))
            for path in (ws / f"artifacts/{name}.joblib", ws / f"artifacts/{name}_freeze.json",
                         ws / f"outputs/runs/{name}/metrics.json",
                         ws / f"outputs/runs/{name}/predictions.csv",
                         ws / f"outputs/runs/{name}/all_class_scores.npz"):
                classical_evidence[str(path.relative_to(ws))] = hash_file(path)
        protocol = {"registered_at": now(), "seeds": list(SEEDS), "training": config,
                    "dataset": dataset, "source_sha256": cnn_source_hash(ws),
                    "classical_source_sha256": _classical_source_hash(ws),
                    "config_sha256": hash_file(config_path), "runner_sha256": hash_file(Path(__file__)),
                    "pretrained_weights": {"url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                                           "sha256": hash_file(weights)},
                    "environment": {**env_info(), "torch": str(torch.__version__),
                                    "torchvision": __import__("torchvision").__version__,
                                    "torch_directml": "0.2.5.dev240914"},
                    "device": {"request": DEVICE, "name": torch_directml.device_name(0).rstrip("\x00"),
                               "float_precision": "float32; no CUDA AMP",
                               "optimizer_note": "aten::lerp.Scalar_out falls back to CPU",
                               "torch_cpu_threads": 8, "opencv_threads": 1},
                    "selection": "session1 fold0 only; 4800 development train / 1200 validation; "
                                 "select stage/epoch by rank1 then lower CE, refit on all 6000 session1",
                    "cnn_test_lock": "freeze all 3 final CNN seeds before evaluating any CNN on session2",
                    "statistics": {"metrics": ["rank1", "rank5", "macro_f1"], "seed_std_ddof": 1,
                                   "never_select_best_test_seed": True, "ensemble": False,
                                   "primary_comparisons": "mean seed Rank-1 versus each of knn, svm, crc",
                                   "paired_bootstrap_unit": "Palm ID, same probe resampling for all models/seeds",
                                   "iterations": 5000, "seed": 42,
                                   "fixed_seed_ci95_and_bonferroni3_familywise95": True,
                                   "seed_std_is_not_a_sampling_CI": True},
                    "cpu_latency": {"query_set": "all 6000 session2 probes per seed", "repetitions": 1,
                                    "threads": 1, "warmup_matching_calls": 3,
                                    "boundary": "read+decode+tensor transforms+network logits; no model loading/ranking"},
                    "test_history": "Classical session2 metrics were already observed in phase 1. "
                                    "CNN settings use existing configs/resnet18.yaml without test-driven changes. "
                                    "This is a sequential/exploratory comparison, not an entirely untouched multi-model test.",
                    "legacy_pilot_warning": "Old HOG pilot string in source has no supporting evidence; "
                                            "user stated no model run before phase-1 benchmark.",
                    "classical_evidence_sha256": classical_evidence}
        atomic_json(registered, protocol)
        for path in sorted((ws / "src/palmprint").rglob("*.py")):
            target = out / "source_snapshot" / path.relative_to(ws)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(config_path, out / "resnet18_registered.yaml")
    else:
        protocol = json.loads(registered.read_text(encoding="utf-8"))
        assert protocol["source_sha256"] == cnn_source_hash(ws)
        assert protocol["config_sha256"] == hash_file(config_path)

    for seed in SEEDS:
        if not (ws / f"artifacts/resnet18_seed{seed}_freeze.json").exists():
            stage(f"train_seed{seed}", lambda seed=seed: train_one(ws, seed=seed, **config))

    lock_path = ws / "artifacts/cnn_protocol_lock.json"
    if not lock_path.exists():
        if any((ws / f"outputs/runs/resnet18_seed{s}/metrics.json").exists() for s in SEEDS):
            raise RuntimeError("CNN test occurred before group lock")
        group = {"locked_at": now(), "seeds": list(SEEDS), "source_sha256": cnn_source_hash(ws),
                 "protocol_sha256": hash_file(registered), "models": {}}
        for seed in SEEDS:
            path = ws / f"artifacts/resnet18_seed{seed}_freeze.json"
            frozen = json.loads(path.read_text(encoding="utf-8"))
            assert frozen["artifact_sha256"] == hash_file(ws / f"artifacts/resnet18_seed{seed}.pth")
            assert frozen["manifest_sha256"] == dataset["manifest_sha256"]
            group["models"][str(seed)] = {"freeze_sha256": hash_file(path),
                                         "artifact_sha256": frozen["artifact_sha256"],
                                         "selected_stage": frozen["selected_stage"],
                                         "head_epochs": frozen["head_epochs_final"],
                                         "layer4_epochs": frozen["finetune_epochs_final"]}
        atomic_json(lock_path, group)
        print(f"ALL THREE CNN SEEDS LOCKED: {lock_path}", flush=True)
    group = json.loads(lock_path.read_text(encoding="utf-8"))
    assert group["source_sha256"] == cnn_source_hash(ws)
    for seed in SEEDS:
        assert group["models"][str(seed)]["freeze_sha256"] == hash_file(ws / f"artifacts/resnet18_seed{seed}_freeze.json")
        if not (ws / f"outputs/runs/resnet18_seed{seed}/metrics.json").exists():
            stage(f"evaluate_seed{seed}", lambda seed=seed:
                  evaluate_one(ws, seed=seed, batch_size=conf["batch_size"], device=DEVICE))

    stage("verify_six_models_and_generate_report", lambda: generate_report(ws))
    rows, prediction_arrays = [], []
    for seed in SEEDS:
        run = ws / f"outputs/runs/resnet18_seed{seed}"
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        f = json.loads((ws / f"artifacts/resnet18_seed{seed}_freeze.json").read_text(encoding="utf-8"))
        rows.append({"seed": seed, "rank1": m["rank1"], "rank5": m["rank5"], "macro_f1": m["macro_f1"],
                     "rank1_ci95_lower": m["rank1_ci95_palm_bootstrap"][0],
                     "rank1_ci95_upper": m["rank1_ci95_palm_bootstrap"][1],
                     "n_probes": m["n"], "correct_probes": int(round(m["rank1"] * m["n"])),
                     "selected_stage": f["selected_stage"], "head_epochs_final": f["head_epochs_final"],
                     "finetune_epochs_final": f["finetune_epochs_final"],
                     "dev_val_best_rank1": f["dev_val_best_rank1"],
                     "artifact_sha256": f["artifact_sha256"], "device": m["device"]})
        pred = pd.read_csv(run / "predictions.csv")
        assert pred.sample_uid.tolist() == test.sample_uid.tolist()
        assert pred.true_palm_id.tolist() == test.palm_id.tolist()
        prediction_arrays.append(pred.pred_palm_id.to_numpy())
    frames = pd.DataFrame(rows)
    tables = ws / "reports/tables"
    frames.to_csv(tables / "cnn_seed_results.csv", index=False)
    aggregate = [{"metric": key, "mean": float(frames[key].mean()),
                  "sample_std_ddof1": float(frames[key].std(ddof=1)), "n_seeds": len(SEEDS)}
                 for key in ("rank1", "rank5", "macro_f1")]
    pd.DataFrame(aggregate).to_csv(tables / "cnn_seed_summary.csv", index=False)
    paired, seed_pairs = [], []
    predictions = np.stack(prediction_arrays)
    truth = test.palm_id.to_numpy()
    for name in ("knn", "svm", "crc"):
        base = pd.read_csv(ws / f"outputs/runs/{name}/predictions.csv").set_index("sample_uid")
        base = base.loc[test.sample_uid.tolist()]
        assert base.true_palm_id.tolist() == truth.tolist()
        base_pred = base.pred_palm_id.to_numpy()
        paired.append({"classical_model": name, **paired_seed_mean_delta(truth, predictions, base_pred)})
        for index, seed in enumerate(SEEDS):
            interval = bootstrap_delta(truth, predictions[index], base_pred)
            seed_pairs.append({"cnn_seed": seed, "classical_model": name,
                               "delta_rank1": float((predictions[index] == truth).mean() - (base_pred == truth).mean()),
                               "ci95_lower": interval[0], "ci95_upper": interval[1],
                               "n_paired_probes": len(truth), "note": "per-seed descriptive 95%, unadjusted"})
    pd.DataFrame(paired).to_csv(tables / "cnn_mean_vs_classical_paired.csv", index=False)
    pd.DataFrame(seed_pairs).to_csv(tables / "cnn_each_seed_vs_classical_paired.csv", index=False)
    atomic_json(out / "recognition_results.json", {"completed_at": now(), "seeds": rows,
                "aggregate": aggregate, "paired_mean_comparisons": paired,
                "source_unchanged_since_registration": cnn_source_hash(ws) == protocol["source_sha256"]})
    print("CNN RECOGNITION RESULTS", flush=True)
    print(frames.to_string(index=False), flush=True)
    print(pd.DataFrame(aggregate).to_string(index=False), flush=True)

    for seed in SEEDS:
        run = ws / f"outputs/runs/resnet18_seed{seed}"
        dest = run / "benchmark_cpu.json"
        if not dest.exists():
            stage(f"cpu_latency_seed{seed}", lambda seed=seed: cpu_latency(ws, seed, test, root))

    for path, expected_hash in protocol["classical_evidence_sha256"].items():
        if hash_file(ws / path) != expected_hash:
            raise RuntimeError(f"Classical evidence changed: {path}")
    atomic_json(out / "run_complete.json", {"completed_at": now(), "seeds": list(SEEDS),
                "source_sha256": cnn_source_hash(ws), "classical_evidence_preserved": True,
                "protocol_sha256": hash_file(registered), "group_lock_sha256": hash_file(lock_path)})
    print("THREE-SEED CNN EXPERIMENT COMPLETED", flush=True)


if __name__ == "__main__":
    main()
