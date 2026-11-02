"""EDA, error diagnostics, CPU benchmark and pre-registered synthetic perturbations."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.dataset import read_gray, resolve_path
from ..data.preparation import load_manifest
from ..features.compcode import CompCodeConfig, CompCodeExtractor
from ..utils import atomic_json
from .metrics import compute_metrics, ranking

# Configure clean typography with native Vietnamese support and no clipping
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def data_eda(ws: Path) -> dict:
    df, _ = load_manifest(ws)
    out = ws / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    tables = ws / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pivot = df.groupby(["session", "palm_id"]).size().unstack(0, fill_value=0)
    pivot.to_csv(tables / "images_per_palm_session.csv")

    colors = {"session1": "#2b5c8f", "session2": "#d95f02"}
    names = {"session1": "Session 1 (Gallery)", "session2": "Session 2 (Probe)"}

    # 1. Unified 1x3 distribution panel for report (300 DPI)
    fig_all, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    keys_labels = [
        ("brightness", "Cường độ sáng trung bình (Mean Intensity)", [0, 255]),
        ("contrast", "Độ tương phản (Pixel Standard Deviation)", None),
        ("blur_proxy", "Độ sắc nét (Biến sai toán tử Laplacian)", None),
    ]

    for idx, (key, label, xlim) in enumerate(keys_labels):
        ax = axes[idx]
        for s_name, group in df.groupby("session"):
            vals = group[key].to_numpy()
            m = float(np.mean(vals))
            ax.hist(
                vals,
                bins=32,
                alpha=0.6,
                color=colors[s_name],
                label=f"{names[s_name]} (TB: {m:.1f})",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.axvline(m, color=colors[s_name], linestyle="--", linewidth=1.5)
        ax.set_xlabel(label, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Số lượng ảnh ROI", fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8.5, frameon=True)
        if xlim:
            ax.set_xlim(xlim)

    fig_all.suptitle(
        "Phân bố các Thuộc tính Chất lượng Ảnh ROI của Bộ dữ liệu Tongji (Session 1 vs Session 2)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig_all.tight_layout(rect=[0, 0, 1, 0.92])
    fig_all.savefig(out / "eda_quality_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig_all)

    # 2. Individual figures (at 300 DPI) for backward compatibility
    for key, label, _ in keys_labels:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
        for s_name, group in df.groupby("session"):
            vals = group[key].to_numpy()
            m = float(np.mean(vals))
            ax.hist(
                vals,
                bins=32,
                alpha=0.6,
                color=colors[s_name],
                label=f"{names[s_name]} (TB: {m:.1f})",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.axvline(m, color=colors[s_name], linestyle="--", linewidth=1.5)
        ax.set_xlabel(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Số lượng ảnh ROI", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, frameon=True)
        fig.tight_layout()
        fig.savefig(out / f"eda_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 3. Class balance plot (300 DPI)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    ax.plot(
        pivot.index,
        pivot.session1,
        label="Session 1 (Gallery)",
        color=colors["session1"],
        linewidth=1.5,
    )
    ax.plot(
        pivot.index,
        pivot.session2,
        label="Session 2 (Probe)",
        color=colors["session2"],
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_xlabel("Chỉ số lớp bàn tay (Palm ID 1 đến 600)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Số lượng ảnh / Palm ID", fontsize=10, fontweight="bold")
    ax.set_title(
        "Phân bố số lượng mẫu theo lớp (600 Palm ID cân bằng tuyệt đối: 10 ảnh/phiên)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_ylim(0, 15)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out / "eda_class_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    quality = (
        df.groupby("session")[["brightness", "contrast", "blur_proxy"]]
        .agg(["mean", "std", "min", "max"])
        .round(5)
    )
    quality.to_csv(tables / "eda_quality_summary.csv")
    return {
        "n": len(df),
        "n_palms": int(df.palm_id.nunique()),
        "quality_is_proxy": True,
        "figures": [p.name for p in out.glob("eda_*.png")],
        "privacy": "No biometric image montages exported automatically",
    }


def error_analysis(ws: Path, model: str) -> dict:
    df, _ = load_manifest(ws)
    pred_path = ws / "outputs/runs" / model / "predictions.csv"
    predictions = pd.read_csv(pred_path)
    joined = predictions.merge(
        df[["sample_uid", "palm_id", "brightness", "contrast", "blur_proxy"]],
        how="left",
        on="sample_uid",
        validate="one_to_one",
    )
    if joined.palm_id.isna().any():
        raise RuntimeError("Missing manifest records for prediction IDs")
    if not np.array_equal(joined.true_palm_id.values, joined.palm_id.values):
        raise RuntimeError("Prediction label mismatch against manifest")
    summary = joined.groupby("correct")[["brightness", "contrast", "blur_proxy"]].agg(
        ["count", "mean", "std"]
    )
    out = ws / "outputs/runs" / model
    summary.to_csv(out / "quality_by_correctness.csv")
    confused = (
        joined[joined.correct == 0]
        .groupby(["true_palm_id", "pred_palm_id"])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    confused.to_csv(out / "most_confused_pairs.csv", header=["count"])
    return {
        "model": model,
        "misclassified": int((joined.correct == 0).sum()),
        "accuracy": float(joined.correct.mean()),
        "caution": "Quality proxies are associations, not proven causal explanations",
    }


def benchmark_cpu(ws: Path, model_name: str, *, limit: int = 40) -> dict:
    from ..utils import hash_file

    df, info = load_manifest(ws)
    if not (ws / "outputs/runs" / model_name / "metrics.json").exists():
        raise RuntimeError("Benchmark only after locked final evaluation")
    if limit < 10:
        # Median and P95 with <10 probes would not be interpretable.
        raise ValueError("Benchmark requires at least 10 test queries")
    data = df[df.session == "session2"].head(limit)
    artifact_file = ws / "artifacts" / f"{model_name}.joblib"
    freeze = json.loads((ws / "artifacts" / f"{model_name}_freeze.json").read_text())
    if hash_file(artifact_file) != freeze["artifact_sha256"]:
        raise RuntimeError("Benchmark artifact hash differs from frozen model")
    model_data = joblib.load(artifact_file)
    extractor = CompCodeExtractor(CompCodeConfig(**model_data["feature_config"]))
    model = model_data["model"]
    root = Path(info["data_root"])
    pre, feature, classify = [], [], []
    # Untimed warmup includes extraction, prediction and potential BLAS initialization.
    first_gray = read_gray(resolve_path(root, data.path.iloc[0]), extractor.config.image_size)
    first_feat = extractor.transform(first_gray)
    for _ in range(3):
        model.scores(first_feat[None, :])
    for path in data.path:
        t0 = time.perf_counter()
        gray = read_gray(resolve_path(root, path), extractor.config.image_size)
        t1 = time.perf_counter()
        z = extractor.transform(gray)
        t2 = time.perf_counter()
        model.scores(z[None, :])
        t3 = time.perf_counter()
        pre.append((t1 - t0) * 1000)
        feature.append((t2 - t1) * 1000)
        classify.append((t3 - t2) * 1000)
    result = {
        "model": model_name,
        "n_queries": len(data),
        "device": "CPU",
        "host": platform.processor(),
        "timing": "single-image wall time, model loaded, 3 warmup matching calls",
        "preprocess_median_ms": float(np.median(pre)),
        "feature_median_ms": float(np.median(feature)),
        "classifier_median_ms": float(np.median(classify)),
        "total_median_ms": float(np.median(np.array(pre) + feature + classify)),
        "total_p95_ms": float(np.percentile(np.array(pre) + feature + classify, 95)),
    }
    atomic_json(ws / "outputs/runs" / model_name / "benchmark_cpu.json", result)
    return result


def robustness(ws: Path, model_name: str, *, limit: int | None = None) -> dict:
    """Fixed stress conditions; cannot update parameters and requires final test already run."""
    df, info = load_manifest(ws)
    model_path = ws / "outputs/runs" / model_name / "metrics.json"
    if not model_path.exists():
        raise RuntimeError("Evaluate frozen main test first; robustness is secondary")
    bundle = joblib.load(ws / "artifacts" / f"{model_name}.joblib")
    extractor = CompCodeExtractor(CompCodeConfig(**bundle["feature_config"]))
    model = bundle["model"]
    test = df[df.session == "session2"].copy()
    if limit:
        test = test.head(limit)
    root = Path(info["data_root"])
    variants = ("clean", "brightness_minus15", "brightness_plus15", "blur3", "rotate_plus5")
    rows = []
    for variant in variants:
        representations = []
        for path in test.path:
            img = read_gray(resolve_path(root, path), extractor.config.image_size)
            if variant == "brightness_minus15":
                img = np.uint8(np.clip(img.astype(np.int16) - 15, 0, 255))
            elif variant == "brightness_plus15":
                img = np.uint8(np.clip(img.astype(np.int16) + 15, 0, 255))
            elif variant == "blur3":
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=1)
            elif variant == "rotate_plus5":
                n = extractor.config.image_size
                mat = cv2.getRotationMatrix2D((n / 2, n / 2), 5, 1.0)
                img = cv2.warpAffine(img, mat, (n, n), borderMode=cv2.BORDER_REFLECT_101)
            representations.append(extractor.transform(img))
        X = np.stack(representations)
        scores = np.concatenate(
            [model.scores(b) for b in np.array_split(X, max(1, (len(X) + 31) // 32))]
        )
        order, _ = ranking(scores, model.classes_)
        metric = compute_metrics(test.palm_id.to_numpy(), order, model.classes_)
        rows.append({"variant": variant, "rank1": metric["rank1"], "rank5": metric["rank5"]})
    outfile = ws / "outputs/runs" / model_name / "robustness.csv"
    pd.DataFrame(rows).to_csv(outfile, index=False)
    return {
        "model": model_name,
        "results": rows,
        "condition": "Synthetic perturbations; not real-world smartphone validation",
    }
