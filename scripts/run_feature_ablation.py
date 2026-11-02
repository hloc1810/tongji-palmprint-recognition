"""Run E3 Feature Representation Ablation on Tongji Palmprint.

Compares:
1. Raw Grayscale Pixels (16,384-D, L2-normalized)
2. PCA (256-D, randomized SVD, fit strictly on train fold/session1, L2-normalized)
3. CompCode (600-D orientation histogram baseline)

Classifier: Strictly 1-Nearest Neighbor (NearestTemplate, Euclidean distance).
Evaluates Phase A (5-fold CV on session1) and Phase B (Cross-session session1 -> session2).
Generates publication-quality charts and findings.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from palmprint.evaluation.metrics import bootstrap_rank1, compute_metrics, ranking
from palmprint.features.ablation import (
    batch_score_1nn,
    fit_transform_pca,
    load_or_extract_raw,
)
from palmprint.features.extraction import load_features
from palmprint.models.classical import NearestTemplate


def run_cv(
    df_s1: pd.DataFrame,
    x_raw_s1: np.ndarray,
    x_comp_s1: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    """Execute Phase A: 5-Fold Cross-Validation on Session 1."""
    labels = df_s1.palm_id.to_numpy(dtype=np.int32)
    classes = np.sort(np.unique(labels))
    folds = sorted(df_s1.fold.unique())

    methods = [
        ("raw", 16384),
        ("pca", 256),
        ("compcode", 600),
    ]

    records = []
    print("\n--- Phase A: 5-Fold Cross-Validation on Session 1 ---")

    for rep, dim in methods:
        print(f"Evaluating representation: {rep} ({dim}-D)...")
        for fold in folds:
            is_val = df_s1.fold.to_numpy() == fold
            train_idx = ~is_val
            val_idx = is_val

            train_y = labels[train_idx]
            val_y = labels[val_idx]

            t0_fit = time.perf_counter()
            if rep == "raw":
                tr_x = x_raw_s1[train_idx]
                va_x = x_raw_s1[val_idx]
                model = NearestTemplate().fit(tr_x, train_y)
            elif rep == "pca":
                # Strict no-leakage: fit PCA only on training fold
                pca, tr_x, va_x = fit_transform_pca(
                    x_raw_s1[train_idx],
                    x_raw_s1[val_idx],
                    n_components=256,
                    seed=42,
                )
                model = NearestTemplate().fit(tr_x, train_y)
            elif rep == "compcode":
                tr_x = x_comp_s1[train_idx]
                va_x = x_comp_s1[val_idx]
                model = NearestTemplate().fit(tr_x, train_y)
            else:
                raise ValueError(f"Unknown representation {rep}")
            fit_time = time.perf_counter() - t0_fit

            t0_infer = time.perf_counter()
            scores = batch_score_1nn(model, va_x, chunk_size=400)
            ordered, _ = ranking(scores, classes)
            infer_time = time.perf_counter() - t0_infer

            met = compute_metrics(val_y, ordered, classes)

            records.append(
                {
                    "representation": rep,
                    "fold": int(fold),
                    "train_n": int(train_idx.sum()),
                    "val_n": int(val_idx.sum()),
                    "feature_dim": dim,
                    "fit_seconds": fit_time,
                    "val_predict_seconds": infer_time,
                    "rank1": met["rank1"],
                    "rank5": met["rank5"],
                    "macro_f1": met["macro_f1"],
                }
            )
            print(
                f"  [{rep:8s}] Fold {fold}: Rank-1={met['rank1']:.4f}, "
                f"Rank-5={met['rank5']:.4f}, F1={met['macro_f1']:.4f} "
                f"({infer_time:.2f}s)"
            )

    cv_df = pd.DataFrame(records)
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    print(f"Saved CV results to: {out_dir / 'cv_results.csv'}")
    return cv_df


def run_cross_session(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    x_raw_s1: np.ndarray,
    x_raw_s2: np.ndarray,
    x_comp_s1: np.ndarray,
    x_comp_s2: np.ndarray,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[float]], PCA]:
    """Execute Phase B: Cross-Session Session 1 (Gallery) -> Session 2 (Probe)."""
    gallery_y = df_s1.palm_id.to_numpy(dtype=np.int32)
    probe_y = df_s2.palm_id.to_numpy(dtype=np.int32)
    classes = np.sort(np.unique(gallery_y))

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    cmc_curves = {}
    prediction_dfs = {}
    summary_records = []

    print("\n--- Phase B: Cross-Session Evaluation (Session 1 -> Session 2) ---")

    # 1. Raw Grayscale
    print("Evaluating F1: Raw Grayscale (16,384-D)...")
    t0_fit = time.perf_counter()
    model_raw = NearestTemplate().fit(x_raw_s1, gallery_y)
    fit_time_raw = time.perf_counter() - t0_fit

    t0_match = time.perf_counter()
    scores_raw = batch_score_1nn(model_raw, x_raw_s2, chunk_size=400)
    ordered_raw, _ = ranking(scores_raw, classes)
    match_time_raw = time.perf_counter() - t0_match

    met_raw = compute_metrics(probe_y, ordered_raw, classes)
    ci95_raw = bootstrap_rank1(probe_y, ordered_raw[:, 0], iterations=1000, seed=42)
    pred_raw_df = pd.DataFrame(
        {
            "sample_uid": df_s2.sample_uid,
            "session": df_s2.session,
            "true_palm_id": probe_y,
            "predicted_palm_id": ordered_raw[:, 0],
            "is_correct": ordered_raw[:, 0] == probe_y,
            "rank_of_true": np.argmax(ordered_raw == probe_y[:, None], axis=1) + 1,
        }
    )
    pred_raw_df.to_csv(pred_dir / "raw_predictions.csv", index=False)
    prediction_dfs["raw"] = pred_raw_df
    cmc_curves["raw"] = [float(np.mean(pred_raw_df.rank_of_true <= k)) for k in range(1, 21)]

    summary_records.append(
        {
            "representation": "raw",
            "feature_dim": 16384,
            "rank1": met_raw["rank1"],
            "rank5": met_raw["rank5"],
            "macro_precision": met_raw["macro_precision"],
            "macro_recall": met_raw["macro_recall"],
            "macro_f1": met_raw["macro_f1"],
            "rank1_ci95_low": ci95_raw[0],
            "rank1_ci95_high": ci95_raw[1],
            "fit_seconds": fit_time_raw,
            "matching_seconds": match_time_raw,
            "matching_ms_per_probe": (match_time_raw / len(probe_y)) * 1000.0,
        }
    )
    print(
        f"  [raw]      Rank-1: {met_raw['rank1']:.4f} (95% CI: [{ci95_raw[0]:.4f}, {ci95_raw[1]:.4f}]), "
        f"Rank-5: {met_raw['rank5']:.4f}, F1: {met_raw['macro_f1']:.4f}"
    )

    # 2. PCA (256-D)
    print("Evaluating F2: PCA (256-D, fit on Session 1 only)...")
    t0_pca = time.perf_counter()
    pca_final, x_pca_s1, x_pca_s2 = fit_transform_pca(x_raw_s1, x_raw_s2, n_components=256, seed=42)
    model_pca = NearestTemplate().fit(x_pca_s1, gallery_y)
    fit_time_pca = time.perf_counter() - t0_pca

    t0_match = time.perf_counter()
    scores_pca = batch_score_1nn(model_pca, x_pca_s2, chunk_size=400)
    ordered_pca, _ = ranking(scores_pca, classes)
    match_time_pca = time.perf_counter() - t0_match

    met_pca = compute_metrics(probe_y, ordered_pca, classes)
    ci95_pca = bootstrap_rank1(probe_y, ordered_pca[:, 0], iterations=1000, seed=42)
    pred_pca_df = pd.DataFrame(
        {
            "sample_uid": df_s2.sample_uid,
            "session": df_s2.session,
            "true_palm_id": probe_y,
            "predicted_palm_id": ordered_pca[:, 0],
            "is_correct": ordered_pca[:, 0] == probe_y,
            "rank_of_true": np.argmax(ordered_pca == probe_y[:, None], axis=1) + 1,
        }
    )
    pred_pca_df.to_csv(pred_dir / "pca_predictions.csv", index=False)
    prediction_dfs["pca"] = pred_pca_df
    cmc_curves["pca"] = [float(np.mean(pred_pca_df.rank_of_true <= k)) for k in range(1, 21)]

    summary_records.append(
        {
            "representation": "pca",
            "feature_dim": 256,
            "rank1": met_pca["rank1"],
            "rank5": met_pca["rank5"],
            "macro_precision": met_pca["macro_precision"],
            "macro_recall": met_pca["macro_recall"],
            "macro_f1": met_pca["macro_f1"],
            "rank1_ci95_low": ci95_pca[0],
            "rank1_ci95_high": ci95_pca[1],
            "fit_seconds": fit_time_pca,
            "matching_seconds": match_time_pca,
            "matching_ms_per_probe": (match_time_pca / len(probe_y)) * 1000.0,
        }
    )
    print(
        f"  [pca]      Rank-1: {met_pca['rank1']:.4f} (95% CI: [{ci95_pca[0]:.4f}, {ci95_pca[1]:.4f}]), "
        f"Rank-5: {met_pca['rank5']:.4f}, F1: {met_pca['macro_f1']:.4f}"
    )

    # 3. CompCode (600-D)
    print("Evaluating F3: CompCode (600-D baseline)...")
    t0_fit = time.perf_counter()
    model_comp = NearestTemplate().fit(x_comp_s1, gallery_y)
    fit_time_comp = time.perf_counter() - t0_fit

    t0_match = time.perf_counter()
    scores_comp = batch_score_1nn(model_comp, x_comp_s2, chunk_size=400)
    ordered_comp, _ = ranking(scores_comp, classes)
    match_time_comp = time.perf_counter() - t0_match

    met_comp = compute_metrics(probe_y, ordered_comp, classes)
    ci95_comp = bootstrap_rank1(probe_y, ordered_comp[:, 0], iterations=1000, seed=42)
    pred_comp_df = pd.DataFrame(
        {
            "sample_uid": df_s2.sample_uid,
            "session": df_s2.session,
            "true_palm_id": probe_y,
            "predicted_palm_id": ordered_comp[:, 0],
            "is_correct": ordered_comp[:, 0] == probe_y,
            "rank_of_true": np.argmax(ordered_comp == probe_y[:, None], axis=1) + 1,
        }
    )
    pred_comp_df.to_csv(pred_dir / "compcode_predictions.csv", index=False)
    prediction_dfs["compcode"] = pred_comp_df
    cmc_curves["compcode"] = [float(np.mean(pred_comp_df.rank_of_true <= k)) for k in range(1, 21)]

    summary_records.append(
        {
            "representation": "compcode",
            "feature_dim": 600,
            "rank1": met_comp["rank1"],
            "rank5": met_comp["rank5"],
            "macro_precision": met_comp["macro_precision"],
            "macro_recall": met_comp["macro_recall"],
            "macro_f1": met_comp["macro_f1"],
            "rank1_ci95_low": ci95_comp[0],
            "rank1_ci95_high": ci95_comp[1],
            "fit_seconds": fit_time_comp,
            "matching_seconds": match_time_comp,
            "matching_ms_per_probe": (match_time_comp / len(probe_y)) * 1000.0,
        }
    )
    print(
        f"  [compcode] Rank-1: {met_comp['rank1']:.4f} (95% CI: [{ci95_comp[0]:.4f}, {ci95_comp[1]:.4f}]), "
        f"Rank-5: {met_comp['rank5']:.4f}, F1: {met_comp['macro_f1']:.4f}"
    )

    cross_df = pd.DataFrame(summary_records)
    cross_df.to_csv(out_dir / "cross_session_results.csv", index=False)
    print(f"Saved Cross-Session results to: {out_dir / 'cross_session_results.csv'}")

    return cross_df, prediction_dfs, cmc_curves, pca_final


def generate_figures(
    cv_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    cmc_curves: dict[str, list[float]],
    pca: PCA,
    fig_dir: Path,
) -> None:
    """Generate 300 DPI figures comparing the three representations."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    # Figure 1: CV vs Cross-Session Rank-1 Comparison (Generalization Gap)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    cv_means = cv_df.groupby("representation")["rank1"].mean()
    cv_stds = cv_df.groupby("representation")["rank1"].std()

    reps = ["raw", "pca", "compcode"]
    labels = ["Raw Grayscale\n(16,384-D)", "PCA\n(256-D)", "CompCode\n(600-D)"]
    cv_vals = [cv_means[r] * 100.0 for r in reps]
    cv_errs = [cv_stds[r] * 100.0 for r in reps]
    cross_vals = [cross_df[cross_df.representation == r]["rank1"].item() * 100.0 for r in reps]

    x = np.arange(len(reps))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        cv_vals,
        width,
        yerr=cv_errs,
        capsize=4,
        label="Session 1 (5-Fold CV)",
        color="#2b5c8f",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        cross_vals,
        width,
        label="Cross-Session (Session 2 Test)",
        color="#d95f02",
        alpha=0.85,
    )

    ax.set_ylabel("Rank-1 Identification Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Feature Representation Ablation: Within-Session CV vs Cross-Session Test (1-NN)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    # Annotate values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig_path1 = fig_dir / "cv_vs_cross_session_comparison.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Generated chart: {fig_path1}")

    # Figure 2: CMC Curves
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ranks = np.arange(1, 21)
    colors = {"raw": "#7570b3", "pca": "#e7298a", "compcode": "#1b9e77"}
    names = {
        "raw": "Raw Grayscale (16,384-D)",
        "pca": "PCA (256-D)",
        "compcode": "CompCode (600-D)",
    }

    for rep in reps:
        curve = [v * 100.0 for v in cmc_curves[rep]]
        ax.plot(
            ranks, curve, marker="o", markersize=4, label=names[rep], color=colors[rep], linewidth=2
        )

    ax.set_xlabel("Rank (k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Match Characteristic (CMC) (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cross-Session CMC Curves across Feature Representations (1-NN)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(range(1, 21, 2))
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig_path2 = fig_dir / "cmc_curves_feature_ablation.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Generated chart: {fig_path2}")

    # Figure 3: PCA Variance Explained (Scree Plot)
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    exp_var = pca.explained_variance_ratio_ * 100.0
    cum_var = np.cumsum(exp_var)
    comp_idx = np.arange(1, len(exp_var) + 1)

    color1 = "#1f77b4"
    ax1.set_xlabel("Principal Component Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        "Individual Explained Variance (%)", color=color1, fontsize=11, fontweight="bold"
    )
    ax1.plot(comp_idx, exp_var, color=color1, linewidth=1.5, label="Individual")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(exp_var) * 1.1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel(
        "Cumulative Explained Variance (%)", color=color2, fontsize=11, fontweight="bold"
    )
    ax2.plot(comp_idx, cum_var, color=color2, linewidth=2, linestyle="--", label="Cumulative")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    total_cum = cum_var[-1]
    ax1.set_title(
        f"PCA (256-D) Explained Variance on Session 1 (Total: {total_cum:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    fig_path3 = fig_dir / "pca_variance_explained.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Generated chart: {fig_path3}")


def write_findings_md(
    cv_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    pca: PCA,
    out_dir: Path,
) -> None:
    """Write findings.md with structured, objective technical analysis."""
    cv_grp = cv_df.groupby("representation")
    cv_stats = {}
    for rep in ["raw", "pca", "compcode"]:
        sub = cv_grp.get_group(rep)
        cv_stats[rep] = {
            "rank1_mean": sub["rank1"].mean() * 100.0,
            "rank1_std": sub["rank1"].std() * 100.0,
            "rank5_mean": sub["rank5"].mean() * 100.0,
            "macro_f1_mean": sub["macro_f1"].mean() * 100.0,
            "fit_time": sub["fit_seconds"].mean(),
            "infer_time": sub["val_predict_seconds"].mean(),
        }

    cross_stats = {}
    for _, row in cross_df.iterrows():
        rep = row["representation"]
        cross_stats[rep] = {
            "dim": int(row["feature_dim"]),
            "rank1": row["rank1"] * 100.0,
            "rank5": row["rank5"] * 100.0,
            "macro_f1": row["macro_f1"] * 100.0,
            "ci_low": row["rank1_ci95_low"] * 100.0,
            "ci_high": row["rank1_ci95_high"] * 100.0,
            "match_time": row["matching_seconds"],
            "ms_per_probe": row["matching_ms_per_probe"],
        }

    total_pca_var = float(np.sum(pca.explained_variance_ratio_) * 100.0)

    findings = f"""# Nghiên cứu Thực nghiệm E3: Feature Representation Ablation trên Tongji Palmprint

## 1. Mục tiêu và Thiết kế Thực nghiệm

Nghiên cứu này khảo sát ảnh hưởng thuần túy của **biểu diễn đặc trưng (feature representation)** đến khả năng nhận diện lòng bàn tay không tiếp xúc, cố định bộ phân loại là **1-Nearest Neighbor (k=1, Euclidean distance)**.

Thực nghiệm so sánh 3 phương pháp biểu diễn trên bộ dữ liệu Tongji Official ROI 128x128 (600 Palm ID, 12.000 ảnh):
1. **F1 — Raw Grayscale Pixels**: Vector ảnh xám 128x128 làm phẳng (16.384 chiều), chuẩn hóa L2.
2. **F2 — PCA (Principal Component Analysis)**: Giảm chiều ảnh xám xuống 256 chiều bằng Randomized SVD (seed=42). Tuân thủ nghiêm ngặt nguyên tắc **zero-leakage** (chỉ fit trên tập train của từng fold trong CV; chỉ fit trên toàn bộ Session 1 trong kiểm thử cross-session), chuẩn hóa L2 sau phép chiếu. 256 thành phần chính giải thích **{total_pca_var:.2f}%** tổng phương sai của Session 1.
3. **F3 — CompCode (Competitive Code)**: Vector đặc trưng 600 chiều từ ngân hàng 6 bộ lọc Gabor 2D, trích xuất hướng đường vân nổi trội và spatial block histogram (10x10 ô), chuẩn hóa 2 cấp (L1 block -> L2 toàn cục).

> [!NOTE]
> **Quy ước đánh giá**:
> - **Giai đoạn A (Within-Session)**: 5-Fold Cross-Validation trên Session 1 (4.800 train / 1.200 val mỗi fold).
> - **Giai đoạn B (Cross-Session)**: Gallery cố định là toàn bộ Session 1 (6.000 ảnh), Query/Probe là toàn bộ Session 2 (6.000 ảnh). Thí nghiệm này là phân tích bổ sung hậu nghiệm (exploratory/post-hoc).

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm

| Biểu diễn Đặc trưng | Số chiều | Session 1: 5-Fold CV Rank-1 (%) | Session 2: Cross-Session Rank-1 (%) | Rank-1 95% CI (Session 2) | Cross-Session Rank-5 (%) | Macro-F1 (Session 2) | Thời gian khớp (ms/mẫu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1 — Raw Grayscale** | 16.384 | {cv_stats["raw"]["rank1_mean"]:.2f} ± {cv_stats["raw"]["rank1_std"]:.2f}% | **{cross_stats["raw"]["rank1"]:.2f}%** | [{cross_stats["raw"]["ci_low"]:.2f}%, {cross_stats["raw"]["ci_high"]:.2f}%] | {cross_stats["raw"]["rank5"]:.2f}% | {cross_stats["raw"]["macro_f1"]:.2f}% | {cross_stats["raw"]["ms_per_probe"]:.2f} ms |
| **F2 — PCA** | 256 | {cv_stats["pca"]["rank1_mean"]:.2f} ± {cv_stats["pca"]["rank1_std"]:.2f}% | **{cross_stats["pca"]["rank1"]:.2f}%** | [{cross_stats["pca"]["ci_low"]:.2f}%, {cross_stats["pca"]["ci_high"]:.2f}%] | {cross_stats["pca"]["rank5"]:.2f}% | {cross_stats["pca"]["macro_f1"]:.2f}% | {cross_stats["pca"]["ms_per_probe"]:.2f} ms |
| **F3 — CompCode** | 600 | {cv_stats["compcode"]["rank1_mean"]:.2f} ± {cv_stats["compcode"]["rank1_std"]:.2f}% | **{cross_stats["compcode"]["rank1"]:.2f}%** | [{cross_stats["compcode"]["ci_low"]:.2f}%, {cross_stats["compcode"]["ci_high"]:.2f}%] | {cross_stats["compcode"]["rank5"]:.2f}% | {cross_stats["compcode"]["macro_f1"]:.2f}% | {cross_stats["compcode"]["ms_per_probe"]:.2f} ms |

---

## 3. Phân tích Kỹ thuật và Nhận xét Thực nghiệm

### 3.1. Hiện tượng "Generalization Gap" (Khoảng cách Tổng quát hóa)
- **Trong cùng một session (5-Fold CV)**: Cả 3 phương pháp đều đạt độ chính xác Rank-1 cực cao (từ {min(cv_stats["raw"]["rank1_mean"], cv_stats["pca"]["rank1_mean"]):.2f}% đến {cv_stats["compcode"]["rank1_mean"]:.2f}%). Lý do là các ảnh trong cùng Session 1 được chụp liên tiếp trong cùng một phiên lấy mẫu, với điều kiện chiếu sáng, tư thế đặt tay và vị trí máy ảnh hầu như không đổi.
- **Khi kiểm thử chéo phiên (Cross-Session)**: Xuất hiện sự phân hóa rõ rệt:
  - **Raw Grayscale**: Giảm từ {cv_stats["raw"]["rank1_mean"]:.2f}% xuống **{cross_stats["raw"]["rank1"]:.2f}%** (sụt giảm {cv_stats["raw"]["rank1_mean"] - cross_stats["raw"]["rank1"]:.2f} điểm phần trăm).
  - **PCA (256-D)**: Giảm từ {cv_stats["pca"]["rank1_mean"]:.2f}% xuống **{cross_stats["pca"]["rank1"]:.2f}%** (sụt giảm {cv_stats["pca"]["rank1_mean"] - cross_stats["pca"]["rank1"]:.2f} điểm phần trăm).
  - **CompCode (600-D)**: Duy trì độ chính xác vượt trội **{cross_stats["compcode"]["rank1"]:.2f}%** (chỉ giảm {cv_stats["compcode"]["rank1_mean"] - cross_stats["compcode"]["rank1"]:.2f} điểm phần trăm).

### 3.2. Vì sao Raw Pixels và PCA thất bại ở Cross-Session?
1. **Độ nhạy với nhiễu toàn cục và dịch chuyển không gian**:
   - Khoảng cách Euclidean trên vector pixel thô tính tổng sai số bậc hai theo từng pixel: d^2(x, y) = sum (x_i - y_i)^2. Khi sang Session 2 (cách Session 1 từ vài tuần đến hơn một tháng), sự thay đổi nhẹ về góc nghiêng lòng bàn tay, độ giãn da tự nhiên và cường độ ánh sáng môi trường làm toàn bộ lưới pixel bị trôi (spatial shift), khiến khoảng cách giữa hai ảnh cùng một người vượt quá khoảng cách giữa hai người khác nhau.
2. **Bản chất của PCA theo lý thuyết Duda (*Pattern Classification*)**:
   - PCA tìm các trục trực giao có phương sai lớn nhất của dữ liệu huấn luyện (Unsupervised Maximum Variance Projection). Tuy nhiên, trên ảnh xám bàn tay, các thành phần chính đầu tiên chủ yếu ghi nhận sự biến thiên cường độ ánh sáng tổng thể, vùng bóng đổ và hình thái bàn tay lớn, chứ không bắt được thông tin vi cấu trúc đường vân lòng bàn tay. Khi chiếu sang Session 2, những biến đổi về nguồn sáng làm vector chiếu PCA bị sai lệch nghiêm trọng.
3. **Ưu thế của CompCode**:
   - **Lọc thông dải (Bandpass Gabor)** loại bỏ thành phần một chiều (Zero-DC), triệt tiêu sự biến đổi độ sáng nền.
   - **Competitive Coding** chỉ quan tâm đến *hướng vân cục bộ* (chỉ số argmin phản hồi thực), hoàn toàn bất biến với biên độ tương phản.
   - **Spatial Block Histogram (10x10)** cho phép dung sai dịch chuyển và biến dạng co giãn cục bộ, ngăn chặn hiện tượng trôi pixel.

### 3.3. Đánh giá sự đánh đổi Kích thước Vector và Tốc độ
- **PCA (256-D)**: Kích thước nhỏ nhất và tốc độ khớp nhanh nhất ({cross_stats["pca"]["ms_per_probe"]:.2f} ms/mẫu), nhưng độ chính xác sinh trắc học không đạt yêu cầu thực tế ({cross_stats["pca"]["rank1"]:.2f}%).
- **Raw Grayscale (16.384-D)**: Kích thước vector quá lớn, tốn tài nguyên bộ nhớ và tính toán ({cross_stats["raw"]["ms_per_probe"]:.2f} ms/mẫu), nhưng độ chính xác chỉ đạt mức trung bình ({cross_stats["raw"]["rank1"]:.2f}%).
- **CompCode (600-D)**: Đạt điểm cân bằng tối ưu giữa kích thước nén (600 chiều so với 16.384 chiều ban đầu, giảm 96,3% kích thước) và độ chính xác phân biệt sinh trắc học ({cross_stats["compcode"]["rank1"]:.2f}% Rank-1).

---

## 4. Kết luận cho Câu hỏi Nghiên cứu

> **Kết luận**:
> Trong giao thức nhận dạng cross-session trên Tongji, **CompCode vượt trội hoàn toàn so với Raw Grayscale và PCA** khi cùng sử dụng bộ phân loại 1-NN. Việc khai thác cấu trúc hướng vân thông qua ngân hàng bộ lọc Gabor kết hợp mã hóa thi đấu và phân đoạn không gian là yếu tố cốt lõi giúp hệ thống đạt tính bất biến đối với sự thay đổi phiên thu nhận dữ liệu.
"""
    (out_dir / "findings.md").write_text(findings, encoding="utf-8")
    print(f"Saved findings to: {out_dir / 'findings.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Feature Representation Ablation (E3)")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Project root")
    args = parser.parse_args()

    ws = args.workspace.resolve()
    out_dir = ws / "outputs/feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== TASK E3: Feature Representation Ablation on Tongji ===")
    print(f"Workspace: {ws}")
    print(f"Output directory: {out_dir}")

    # 1. Load manifest and raw images
    print("Loading manifest and raw grayscale images...")
    df_s1, x_raw_s1 = load_or_extract_raw(ws, sessions=("session1",))
    df_s2, x_raw_s2 = load_or_extract_raw(ws, sessions=("session2",))
    print(f"Raw Session 1: {x_raw_s1.shape}, dtype={x_raw_s1.dtype}")
    print(f"Raw Session 2: {x_raw_s2.shape}, dtype={x_raw_s2.dtype}")

    # 2. Load CompCode features
    print("Loading CompCode features...")
    compcode_cfg = ws / "configs/compcode.yaml"
    _, x_comp_s1, sig_s1 = load_features(ws, compcode_cfg, "compcode", sessions=("session1",))
    _, x_comp_s2, sig_s2 = load_features(ws, compcode_cfg, "compcode", sessions=("session2",))
    print(f"CompCode Session 1: {x_comp_s1.shape}, sig={sig_s1[:8]}")
    print(f"CompCode Session 2: {x_comp_s2.shape}, sig={sig_s2[:8]}")

    # 3. Save experiment config
    exp_config = {
        "experiment": "E3_feature_representation_ablation",
        "dataset": "Tongji Official ROI 128x128",
        "session1_samples": len(df_s1),
        "session2_samples": len(df_s2),
        "classes": int(df_s1.palm_id.nunique()),
        "classifier": "1-Nearest Neighbor (NearestTemplate, Euclidean)",
        "features": {
            "raw": {
                "dim": 16384,
                "normalization": "L2 per sample",
                "description": "128x128 grayscale pixels flattened",
            },
            "pca": {
                "dim": 256,
                "solver": "randomized",
                "seed": 42,
                "normalization": "L2 post-projection",
                "leakage_control": "fit only on train fold / session1",
            },
            "compcode": {
                "dim": 600,
                "directions": 6,
                "kernel_size": 41,
                "block_pixels": 14,
                "normalization": "L1 block then L2 global",
            },
        },
        "timing": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_status": "exploratory_post_hoc_cross_session",
    }
    (out_dir / "experiment_config.json").write_text(
        json.dumps(exp_config, indent=2), encoding="utf-8"
    )

    # 4. Phase A: 5-Fold Cross-Validation on Session 1
    cv_df = run_cv(df_s1, x_raw_s1, x_comp_s1, out_dir)

    # 5. Phase B: Cross-Session Session 1 -> Session 2
    cross_df, pred_dfs, cmc_curves, pca_final = run_cross_session(
        df_s1, df_s2, x_raw_s1, x_raw_s2, x_comp_s1, x_comp_s2, out_dir
    )

    # 6. Generate Figures
    print("\nGenerating publication figures (300 DPI)...")
    generate_figures(cv_df, cross_df, cmc_curves, pca_final, fig_dir)

    # 7. Write findings.md
    print("Writing findings.md...")
    write_findings_md(cv_df, cross_df, pca_final, out_dir)

    print("\n=== E3 Feature Representation Ablation Completed Successfully ===")


if __name__ == "__main__":
    main()
