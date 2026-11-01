"""Resume only postprocessing/timing after a float32 derived-score-gap precision issue.

Original model/source/metrics/prediction/score artifacts remain unchanged. A separate
verification view canonicalizes only the derived gap after bounding float32 error.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from run_experiments import SEEDS, cnn_source_hash, cpu_latency
from palmprint.data.preparation import load_manifest
from palmprint.evaluation.metrics import bootstrap_delta
from palmprint.reporting.cnn_comparison import paired_seed_mean_delta
from palmprint.reporting.summary import generate_report
from palmprint.utils import atomic_json, hash_file


def main():
    ws = Path.cwd()
    out = ws / "outputs/cnn_experiments"
    protocol = json.loads((out / "protocol_before_cnn_test.json").read_text(encoding="utf-8"))
    assert cnn_source_hash(ws) == protocol["source_sha256"]
    frame, info = load_manifest(ws)
    test = frame[frame.session == "session2"].reset_index(drop=True)
    view = out / "verification_view"
    if view.exists():
        raise RuntimeError("Verification view exists; inspect prior postprocessing before repeating")
    (view / "data/metadata").mkdir(parents=True)
    shutil.copy2(ws / "data/metadata/manifest.csv", view / "data/metadata/manifest.csv")
    shutil.copy2(ws / "data/metadata/local_dataset.json", view / "data/metadata/local_dataset.json")
    shutil.copytree(ws / "artifacts", view / "artifacts")
    shutil.copytree(ws / "outputs/cv", view / "outputs/cv")
    audit = []
    for name in ("knn", "svm", "crc", *(f"resnet18_seed{s}" for s in SEEDS)):
        original = ws / f"outputs/runs/{name}"
        derived = view / f"outputs/runs/{name}"
        shutil.copytree(original, derived)
        pred = pd.read_csv(original / "predictions.csv")
        with np.load(original / "all_class_scores.npz", allow_pickle=False) as data:
            values = np.sort(data["scores"], axis=1)
        exact = values[:, 1] - values[:, 0]
        observed = pred.score_gap_top2_minus_top1.to_numpy()
        error = np.abs(exact - observed)
        bound = np.finfo(np.float32).eps * np.maximum(np.abs(exact), 1.0)
        if not (error <= bound).all():
            raise RuntimeError(f"Gap error exceeds float32 rounding bound: {name}")
        if name.startswith("resnet18"):
            pred["score_gap_top2_minus_top1"] = exact
            pred.to_csv(derived / "predictions.csv", index=False)
            pred.loc[pred.correct == 0].to_csv(derived / "errors.csv", index=False)
        audit.append({"model": name, "max_absolute_gap_rounding_error": float(error.max()),
                      "rows_with_gap_difference_above_1e_minus8": int((error > 1e-8).sum()),
                      "all_within_float32_rounding_bound": True,
                      "raw_predictions_sha256": hash_file(original / "predictions.csv"),
                      "verification_view_predictions_sha256": hash_file(derived / "predictions.csv"),
                      "raw_scores_sha256": hash_file(original / "all_class_scores.npz")})
    atomic_json(out / "float32_gap_verification_audit.json", {
        "reason": "Writer subtracts float32 ranked scores; strict verifier subtracts saved float64 casts",
        "normalization": "derived view only: recompute gap in float64, after checking float32 ULP bound",
        "original_artifacts_modified": False,
        "rankings_metrics_scores_and_model_selection_changed": False, "runs": audit})
    generate_report(view)
    for folder in ("tables", "figures"):
        for path in (view / f"reports/{folder}").iterdir():
            dest = ws / f"reports/{folder}" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    verified = pd.read_csv(ws / "reports/tables/evidence_verification.csv")
    verified = verified.merge(pd.DataFrame(audit)[["model", "raw_predictions_sha256",
        "verification_view_predictions_sha256", "max_absolute_gap_rounding_error"]], on="model", validate="one_to_one")
    verified["verification_policy"] = "canonical float64 gap view; raw gap bounded at float32 precision"
    verified.to_csv(ws / "reports/tables/evidence_verification.csv", index=False)

    rows, predictions = [], []
    for seed in SEEDS:
        run = ws / f"outputs/runs/resnet18_seed{seed}"
        metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        frozen = json.loads((ws / f"artifacts/resnet18_seed{seed}_freeze.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(run / "predictions.csv")
        assert pred.sample_uid.tolist() == test.sample_uid.tolist()
        assert pred.true_palm_id.tolist() == test.palm_id.tolist()
        predictions.append(pred.pred_palm_id.to_numpy())
        rows.append({"seed": seed, "rank1": metric["rank1"], "rank5": metric["rank5"],
                     "macro_f1": metric["macro_f1"], "rank1_ci95_lower": metric["rank1_ci95_palm_bootstrap"][0],
                     "rank1_ci95_upper": metric["rank1_ci95_palm_bootstrap"][1],
                     "n_probes": len(pred), "correct_probes": int(pred.correct.sum()),
                     "selected_stage": frozen["selected_stage"], "head_epochs_final": frozen["head_epochs_final"],
                     "finetune_epochs_final": frozen["finetune_epochs_final"],
                     "dev_val_best_rank1": frozen["dev_val_best_rank1"],
                     "artifact_sha256": frozen["artifact_sha256"], "device": metric["device"]})
    table = pd.DataFrame(rows)
    table.to_csv(ws / "reports/tables/cnn_seed_results.csv", index=False)
    aggregate = [{"metric": key, "mean": float(table[key].mean()), "sample_std_ddof1": float(table[key].std(ddof=1)),
                  "n_seeds": 3} for key in ("rank1", "rank5", "macro_f1")]
    pd.DataFrame(aggregate).to_csv(ws / "reports/tables/cnn_seed_summary.csv", index=False)
    paired, individual = [], []
    truth = test.palm_id.to_numpy()
    for name in ("knn", "svm", "crc"):
        baseline = pd.read_csv(ws / f"outputs/runs/{name}/predictions.csv").set_index("sample_uid").loc[test.sample_uid]
        assert baseline.true_palm_id.tolist() == truth.tolist()
        base_pred = baseline.pred_palm_id.to_numpy()
        paired.append({"classical_model": name, **paired_seed_mean_delta(truth, np.stack(predictions), base_pred)})
        for seed, prediction in zip(SEEDS, predictions):
            interval = bootstrap_delta(truth, prediction, base_pred)
            individual.append({"cnn_seed": seed, "classical_model": name,
                               "delta_rank1": float((prediction == truth).mean() - (base_pred == truth).mean()),
                               "ci95_lower": interval[0], "ci95_upper": interval[1],
                               "n_paired_probes": len(truth), "note": "per-seed descriptive 95%, unadjusted"})
    pd.DataFrame(paired).to_csv(ws / "reports/tables/cnn_mean_vs_classical_paired.csv", index=False)
    pd.DataFrame(individual).to_csv(ws / "reports/tables/cnn_each_seed_vs_classical_paired.csv", index=False)
    atomic_json(out / "recognition_results.json", {"completed_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
                "seeds": rows, "aggregate": aggregate, "paired_mean_comparisons": paired,
                "source_unchanged_since_registration": cnn_source_hash(ws) == protocol["source_sha256"]})
    print(table.to_string(index=False), flush=True)
    print(pd.DataFrame(aggregate).to_string(index=False), flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    for seed in SEEDS:
        target = ws / f"outputs/runs/resnet18_seed{seed}/benchmark_cpu.json"
        if not target.exists():
            print(f"START CPU LATENCY SEED {seed}", flush=True)
            result = cpu_latency(ws, seed, test, Path(info["data_root"]))
            print(result, flush=True)
    for relative, digest in protocol["classical_evidence_sha256"].items():
        assert hash_file(ws / relative) == digest
    atomic_json(out / "run_complete.json", {"completed_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
                "seeds": list(SEEDS), "source_sha256": cnn_source_hash(ws),
                "classical_evidence_preserved": True, "protocol_sha256": hash_file(out / "protocol_before_cnn_test.json"),
                "group_lock_sha256": hash_file(ws / "artifacts/cnn_protocol_lock.json"),
                "postprocessing": "float32 gap compatibility in a separate verification view",
                "postprocessing_script_sha256": hash_file(Path(__file__))})
    print("COMPLETED THREE-SEED EXPERIMENT INCLUDING POSTPROCESSING", flush=True)


if __name__ == "__main__":
    main()
