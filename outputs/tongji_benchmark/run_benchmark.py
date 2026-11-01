"""Execute the existing, predeclared Tongji classical protocol and record measurements.

Run from the repository root with .venv311/Scripts/python.exe -u
outputs/tongji_benchmark/run_benchmark.py. No model/extractor implementation is changed.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from palmprint.data.dataset import validate_manifest
from palmprint.data.preparation import load_manifest
from palmprint.evaluation.analysis import benchmark_cpu, error_analysis
from palmprint.features.extraction import load_features
from palmprint.reporting.summary import generate_report
from palmprint.training.classical import evaluate_final, final_train, lock_protocol
from palmprint.training.selection import _classical_source_hash, cross_validate
from palmprint.utils import atomic_json, env_info, hash_file, read_yaml


def now():
    return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()


def main():
    ws = Path.cwd()
    out = ws / "outputs/tongji_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    config = ws / "configs/compcode.yaml"
    manifest, info = load_manifest(ws)
    validate_manifest(manifest, folds=5, strict=True)
    assert info["strict"] and info["folds"] == 5 and info["seed"] == 42
    assert manifest.groupby(["session", "palm_id"]).size().eq(10).all()
    cv2.setNumThreads(1)
    warnings.simplefilter("error", ConvergenceWarning)
    events = []

    def stage(name, action):
        started = time.perf_counter()
        event = {"stage": name, "started_at": now(), "status": "running"}
        events.append(event)
        atomic_json(out / "progress.json", events)
        print(f"[{event['started_at']}] START {name}", flush=True)
        try:
            value = action()
        except Exception as exc:
            event.update(status="failed", error=repr(exc), finished_at=now(),
                         wall_seconds=time.perf_counter() - started)
            atomic_json(out / "progress.json", events)
            raise
        event.update(status="completed", finished_at=now(),
                     wall_seconds=time.perf_counter() - started)
        atomic_json(out / "progress.json", events)
        print(f"[{event['finished_at']}] DONE {name}: {event['wall_seconds']:.3f} s", flush=True)
        if isinstance(value, dict):
            print(json.dumps(value, ensure_ascii=False, default=str), flush=True)
        return value

    with threadpool_limits(limits=1):
        prereg_path = out / "protocol_before_test.json"
        if not prereg_path.exists():
            if list((ws / "artifacts").glob("*_freeze.json")):
                raise RuntimeError("Existing frozen artifacts require an explicitly documented continuation")
            prereg = {
                "registered_at": now(), "models": ["knn", "svm", "crc"],
                "data": info, "train_gallery_n": 6000, "test_probe_n": 6000,
                "classes": 600, "cv_folds": 5, "fold_seed": 42,
                "train_per_class_per_fold": 8, "validation_per_class_per_fold": 2,
                "features": read_yaml(config), "model_config": read_yaml(ws / "configs/models.yaml"),
                "selection": "maximize mean session1 CV Rank-1; ties choose smallest parameter",
                "latency": {"n_queries": 6000, "query_set": "all session2, manifest order",
                            "repetitions": 1, "warmup_matching_calls": 3,
                            "mode": "single image, sequential models, loaded model; read+feature+matching",
                            "threads": 1, "model_loading_included": False,
                            "cache_condition": "OS file cache may be warm; feature recomputed per query"},
                "environment": env_info(), "cpu": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "threadpools": threadpool_info(),
                "opencv_threads": cv2.getNumThreads(),
                "source_sha256": _classical_source_hash(ws),
                "runner_sha256": hash_file(Path(__file__)),
                "feature_config_sha256": hash_file(config),
                "model_config_sha256": hash_file(ws / "configs/models.yaml"),
                "session2_policy": "no feature extraction or recognition evaluation until all three models locked",
                "legacy_warning_context": "Source contains a hardcoded historical HOG pilot warning. "
                    "No supporting evidence is present; user states no prior Tongji model run. "
                    "Preserve original source and distinguish this legacy text from measured evidence.",
                "method_scope": "paper-inspired CompCode; MATLAB parity not verified",
            }
            atomic_json(prereg_path, prereg)
            snapshot = out / "source_snapshot"
            for path in sorted((ws / "src/palmprint").rglob("*.py")):
                dest = snapshot / path.relative_to(ws)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            for path in (config, ws / "configs/models.yaml", ws / "configs/dataset.yaml"):
                dest = snapshot / path.relative_to(ws)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        else:
            prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
            if prereg["source_sha256"] != _classical_source_hash(ws):
                raise RuntimeError("Registered implementation changed; cannot continue this benchmark")

        print(f"Registered protocol: {prereg_path}", flush=True)
        if not (ws / "artifacts/protocol_lock.json").exists():
            stage("session1_features", lambda: {
                "samples": len((payload := load_features(ws, config))[0]),
                "feature_shape": list(payload[1].shape), "signature": payload[2]})
            for name in ("knn", "svm", "crc"):
                selected = ws / f"outputs/cv/compcode_{name}_selection.json"
                if not selected.exists():
                    stage(f"cv_{name}", lambda name=name: cross_validate(ws, name, config))
                frozen = ws / f"artifacts/{name}_freeze.json"
                if not frozen.exists():
                    stage(f"train_{name}", lambda name=name: final_train(ws, name, config))
            stage("lock_all_classical_models", lambda: lock_protocol(ws))

        for name in ("knn", "svm", "crc"):
            if not (ws / f"outputs/runs/{name}/metrics.json").exists():
                stage(f"evaluate_{name}", lambda name=name: evaluate_final(ws, name, config))

        comparison = stage("verify_evidence_and_generate_report", lambda: generate_report(ws))
        latency = {}
        for name in ("knn", "svm", "crc"):
            path = ws / f"outputs/runs/{name}/benchmark_cpu.json"
            if path.exists():
                latency[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                latency[name] = stage(f"latency_{name}_all6000", lambda name=name:
                                      benchmark_cpu(ws, name, limit=6000))
            stage(f"error_analysis_{name}", lambda name=name: error_analysis(ws, name))

        records = []
        cv_grid = []
        for name in ("knn", "svm", "crc"):
            metric = json.loads((ws / f"outputs/runs/{name}/metrics.json").read_text(encoding="utf-8"))
            choice = json.loads((ws / f"outputs/cv/compcode_{name}_selection.json").read_text(encoding="utf-8"))
            row = comparison.loc[comparison.model == name].iloc[0].to_dict()
            row.update(best_parameter=choice["best_parameter"], cv_mean_rank1=choice["mean_cv_rank1"],
                       cv_fold_std_rank1=choice["std_cv_rank1"],
                       correct_probes=int(round(metric["rank1"] * metric["n"])),
                       latency_n_queries=latency[name]["n_queries"],
                       latency_median_ms=latency[name]["total_median_ms"],
                       latency_p95_ms=latency[name]["total_p95_ms"],
                       classifier_single_image_median_ms=latency[name]["classifier_median_ms"],
                       artifact_size_bytes=(ws / f"artifacts/{name}.joblib").stat().st_size)
            records.append(row)
            cv = pd.read_csv(ws / f"outputs/cv/compcode_{name}.csv")
            for param, sub in cv.groupby("parameter"):
                cv_grid.append({"model": name, "parameter": param, "folds": len(sub),
                                "mean_rank1": sub.rank1.mean(), "std_rank1": sub.rank1.std(),
                                "mean_rank5": sub.rank5.mean(), "mean_macro_f1": sub.macro_f1.mean(),
                                "selected": bool(param == choice["best_parameter"])})
        result = pd.DataFrame(records)
        table = ws / "reports/tables/tongji_benchmark.csv"
        result.to_csv(table, index=False)
        pd.DataFrame(cv_grid).to_csv(ws / "reports/tables/tongji_cv_grid_summary.csv", index=False)
        atomic_json(out / "benchmark_results.json", {
            "completed_at": now(), "protocol_sha256": hash_file(prereg_path),
            "results": records, "timing_measurements": latency,
            "table_sha256": hash_file(table), "source_unchanged":
                prereg["source_sha256"] == _classical_source_hash(ws)})
        lines = ["# Tongji benchmark — M1/M2/M3", "", f"Completed: {now()}", "",
                 "Measured on 600 Palm IDs: session1 6,000 gallery/train images; session2 6,000 probes.",
                 "Five-fold session1 CV, seed 42. Shared 600-dimensional paper-inspired CompCode histogram.",
                 "", "| Model | Parameter | CV Rank-1 | Test Rank-1 | Test Rank-5 | Macro-F1 | E2E median ms | E2E P95 ms |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in records:
            lines.append(f"| {r['model']} | {r['best_parameter']:g} | {100*r['cv_mean_rank1']:.4f}% | "
                         f"{100*r['rank1']:.4f}% | {100*r['rank5']:.4f}% | {100*r['macro_f1']:.4f}% | "
                         f"{r['latency_median_ms']:.4f} | {r['latency_p95_ms']:.4f} |")
        lines += ["", "E2E timing: CPU, one thread, one image per call, all 6,000 probes, three warmup matching calls;",
                  "includes image read, CompCode extraction and class scoring, excludes model loading. File cache may be warm.",
                  "Batch-amortized classifier timings are separate columns in the CSV.", "",
                  "The old historical-HOG-pilot warning in raw metrics is inherited source text, not a verified experiment history.",
                  "User stated there was no prior Tongji model run. See protocol_before_test.json for this context.",
                  "MATLAB parity and equivalence to the authors' reported accuracy are not claimed.", "",
                  "Evidence: reports/tables/tongji_benchmark.csv, tongji_cv_grid_summary.csv, model_comparison.csv,",
                  "paired_rank1_differences.csv, evidence_verification.csv; outputs/runs/{knn,svm,crc}/;",
                  "artifacts/protocol_lock.json; outputs/tongji_benchmark/protocol_before_test.json and progress.json.", ""]
        (ws / "reports/TONGJI_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
        print("MEASURED TONGJI BENCHMARK", flush=True)
        print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
