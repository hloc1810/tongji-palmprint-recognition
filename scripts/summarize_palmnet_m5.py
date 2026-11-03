"""Audit and summarize completed independent M5 runs; never fit or tune a model."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from palmprint.evaluation.metrics import bootstrap_delta, compute_metrics, ranking
from palmprint.utils import atomic_json, hash_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/palmnet_m5"
NAMES = {
    "nearest_template": "M5 PalmNet + 1-NN (full)",
    "selected_1nn": "PalmNet + Variance-1024 + 1-NN",
    "linear_svm": "PalmNet + Variance-1024 + Linear SVM",
    "crc_rls": "PalmNet + Variance-1024 + CRC_RLS",
}


def main():
    manifest_path = ROOT / "data/metadata/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    probes = manifest[manifest.session == "session2"].sort_values(
        ["palm_id", "sample_uid"], kind="stable"
    )
    frozen = json.loads((ROOT / "artifacts/palmnet_m5/freeze.json").read_text())
    if frozen["manifest_sha256"] != hash_file(manifest_path):
        raise RuntimeError("Manifest changed since M5 fitting.")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    predictions_by_method = {}
    for name, title in NAMES.items():
        folder = OUT / name
        if not (folder / "metrics.json").exists():
            if name == "nearest_template":
                raise RuntimeError("M5 primary benchmark has not completed.")
            continue
        metric = json.loads((folder / "metrics.json").read_text())
        for file, key in [
            ("predictions.csv", "prediction_sha256"),
            ("scores.npz", "scores_sha256"),
        ]:
            if hash_file(folder / file) != metric[key]:
                raise RuntimeError(f"Changed result evidence: {name}/{file}")
        pred = pd.read_csv(folder / "predictions.csv")
        if (
            pred.sample_uid.tolist() != probes.sample_uid.tolist()
            or not np.array_equal(pred.true_palm_id, probes.palm_id)
            or metric["n"] != 6000
            or metric["classes"] != 600
            or metric["manifest_sha256"] != frozen["manifest_sha256"]
            or metric["model_sha256"] != frozen["model_sha256"]
        ):
            raise RuntimeError(f"Wrong UID/label/model alignment: {name}")
        with np.load(folder / "scores.npz", allow_pickle=False) as saved:
            ranked, _ = ranking(saved["scores"], saved["classes"])
            expected = compute_metrics(probes.palm_id.to_numpy(), ranked, saved["classes"])
        for key in ("rank1", "rank5", "macro_f1"):
            if not np.isclose(expected[key], metric[key], atol=1e-12):
                raise RuntimeError(f"Metric does not match score evidence: {name}/{key}")
        if not np.array_equal(ranked[:, 0], pred.pred_rank1):
            raise RuntimeError("Predictions do not match full class scores.")
        predictions_by_method[name] = ranked[:, 0].copy()
        rows.append(
            {
                "model": title,
                "features": metric["feature_dim"],
                "rank1_pct": 100 * metric["rank1"],
                "rank5_pct": 100 * metric["rank5"],
                "macro_f1_pct": 100 * metric["macro_f1"],
                "ci95_low_pct": 100 * metric["rank1_ci95_palm_bootstrap"][0],
                "ci95_high_pct": 100 * metric["rank1_ci95_palm_bootstrap"][1],
                "matching_seconds": metric["matching_seconds"],
            }
        )
        curve = [
            (ranked[:, :k] == probes.palm_id.to_numpy()[:, None]).any(axis=1).mean()
            for k in range(1, 21)
        ]
        for ax in axes:
            ax.plot(range(1, 21), np.asarray(curve) * 100, label=title, linewidth=2)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    frozen["benchmark_package_versions"] = {
        name: version(name)
        for name in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "sparse-dot-topn"]
    }
    previous_path = OUT / "protocol.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    )
    current_hashes = {
        str(path.relative_to(ROOT)): hash_file(path)
        for path in [
            ROOT / "src/palmprint/training/palmnet.py",
            ROOT / "src/palmprint/training/palmnet_classifiers.py",
            ROOT / "src/palmprint/features/palmnet.py",
            ROOT / "src/palmprint/models/palmnet_matching.py",
        ]
    }
    if previous.get("model_sha256") == frozen["model_sha256"]:
        frozen["benchmark_source_hashes"] = previous.get("benchmark_source_hashes", current_hashes)
    else:
        frozen["benchmark_source_hashes"] = current_hashes
    frozen["publication_source_hashes"] = current_hashes
    frozen["publication_note"] = (
        "The experiment source differs from the source at summary time; hashes are retained separately."
        if frozen["benchmark_source_hashes"] != current_hashes
        else "Source hashes were recorded when this summary was produced."
    )
    frozen.pop("data_root", None)
    atomic_json(OUT / "protocol.json", frozen)
    comparisons = {}
    y = probes.palm_id.to_numpy()
    for name in ("nearest_template", "linear_svm", "crc_rls"):
        if name not in predictions_by_method or "selected_1nn" not in predictions_by_method:
            continue
        a, b = predictions_by_method[name], predictions_by_method["selected_1nn"]
        comparisons[name + "_minus_selected_1nn"] = {
            "delta_rank1_pp": 100 * (np.mean(a == y) - np.mean(b == y)),
            "ci95_pp": [100 * v for v in bootstrap_delta(y, a, b, iterations=10000, seed=42)],
            "bootstrap_unit": "Palm ID",
            "iterations": 10000,
            "seed": 42,
        }
    atomic_json(OUT / "paired_comparisons.json", comparisons)
    for ax in axes:
        ax.set(xlabel="Rank k", ylabel="Identification accuracy (%)", xlim=(1, 20))
        ax.set_xticks([1, 2, 5, 10, 15, 20])
        ax.grid(alpha=0.25)
    axes[0].set(ylim=(0, 100.05), title="Full scale")
    axes[1].set(ylim=(98, 100.05), title="Detail: 98-100%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Tongji: Session 1 gallery / Session 2 probe (600 Palm IDs)")
    fig.subplots_adjust(bottom=0.27, top=0.85, wspace=0.3)
    figure = ROOT / "reports/figures/palmnet_m5_cmc.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=300)
    plt.close(fig)
    text = [
        "# PCA-Gabor PalmNet — kết quả đã kiểm tra",
        "",
        "6.000 gallery Session 1 / 6.000 probe Session 2, 600 Palm ID.",
        "",
        "| Phương pháp | Số chiều | Rank-1 (%) | Rank-5 (%) | Macro-F1 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['model']} | {row['features']:,} | {row['rank1_pct']:.2f} | "
            f"{row['rank5_pct']:.2f} | {row['macro_f1_pct']:.2f} |"
        )
    text += [
        "",
        "Các nhánh 1.024 chiều dùng cùng phép chọn phương sai fit trên Session 1.",
        "So sánh SVM/CRC với 1-NN 1.024 chiều để tách tác động classifier.",
        "Đây là cấu hình mở rộng cố định, chưa tuning bằng validation Session 1.",
        "",
        f"Nguồn PalmNet: {frozen['vendor']['commit']}",
        f"Manifest SHA-256: {frozen['manifest_sha256']}",
        f"Model SHA-256: {frozen['model_sha256']}",
        "",
        "Timing là theo batch và gồm IO cache, không phải single-query latency.",
    ]
    (OUT / "findings.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
