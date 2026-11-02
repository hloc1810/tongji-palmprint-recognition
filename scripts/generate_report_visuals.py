"""Generate publication-grade qualitative and supplementary figures for final report.

Outputs:
1. dataset_samples_session1_vs_session2.png
2. compcode_feature_extraction_pipeline.png
3. accuracy_vs_latency_tradeoff.png
4. qualitative_predictions_success_failure.png
"""

import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palmprint.features.compcode import CompCodeConfig, CompCodeExtractor

# Configure clean typography with native Vietnamese support and no clipping
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "reports/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw"
MANIFEST_PATH = ROOT / "data/metadata/manifest.csv"


def safe_savefig(fig: plt.Figure, out_path: Path, **kwargs) -> None:
    """Save figure with retry logic against transient Windows file locks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f"{out_path.stem}_tmp.png")
    fig.savefig(temp_path, **kwargs)
    for _ in range(5):
        try:
            if out_path.exists():
                out_path.unlink()
            temp_path.replace(out_path)
            return
        except (OSError, PermissionError):
            time.sleep(0.1)
    temp_path.replace(out_path)


def plot_dataset_samples():
    """Figure 1: Illustrate intra-class variations across Session 1 and Session 2."""
    manifest = pd.read_csv(MANIFEST_PATH)
    palms = [1, 25, 126, 403]

    fig, axes = plt.subplots(len(palms), 4, figsize=(10, 10.5), dpi=300)
    plt.subplots_adjust(top=0.91, bottom=0.03, left=0.05, right=0.95, wspace=0.15, hspace=0.35)

    for row_idx, palm_id in enumerate(palms):
        s1_imgs = manifest[(manifest.palm_id == palm_id) & (manifest.session == "session1")][
            "path"
        ].tolist()[:2]
        s2_imgs = manifest[(manifest.palm_id == palm_id) & (manifest.session == "session2")][
            "path"
        ].tolist()[:2]

        cols = [
            (s1_imgs[0], f"Palm {palm_id} — S1 (#1)\n(Gallery)"),
            (s1_imgs[1], f"Palm {palm_id} — S1 (#2)\n(Gallery)"),
            (s2_imgs[0], f"Palm {palm_id} — S2 (#1)\n(Probe)"),
            (s2_imgs[1], f"Palm {palm_id} — S2 (#2)\n(Probe)"),
        ]

        for col_idx, (rel_path, title) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            img = cv2.imread(str(RAW / rel_path), cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9, fontweight="bold" if "S2" in title else "normal")
            ax.axis("off")

    fig.suptitle(
        "Tongji Contactless Palmprint: Session 1 (Gallery) vs Session 2 (Probe)\n"
        "Minh họa biến thiên chiếu sáng, góc xoay và độ co giãn nếp vân giữa 2 phiên",
        fontsize=12,
        fontweight="bold",
        y=0.965,
    )

    out_path = FIGDIR / "dataset_samples_session1_vs_session2.png"
    safe_savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def plot_compcode_pipeline():
    """Figure 2: CompCode 600-D feature extraction pipeline."""
    sample_path = RAW / "session1/00001.bmp"
    img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)

    cfg = CompCodeConfig()
    extractor = CompCodeExtractor(cfg)

    # 1. Input ROI
    # 2. 6 Gabor Kernels
    # 3. Orientation Map (0..5)
    # 4. 10x10 spatial grid on ROI
    # 5. 600-D histogram

    # (a) Input ROI, (b) Gabor kernels, (c) Orientation map, (d) 10x10 grid, (e, f) Histograms
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5), dpi=300)
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96, wspace=0.28, hspace=0.38)

    # (a) Input ROI
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("(a) ROI Gốc (128×128)", fontsize=10, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    # (b) 6 Gabor Kernels combined into one visual
    gabor_montage = np.zeros((41 * 2, 41 * 3), dtype=np.float32)
    for idx, k in enumerate(extractor.kernels):
        r_i = idx // 3
        c_i = idx % 3
        gabor_montage[r_i * 41 : (r_i + 1) * 41, c_i * 41 : (c_i + 1) * 41] = k
    axes[0, 1].imshow(gabor_montage, cmap="seismic")
    axes[0, 1].set_title(
        "(b) 6 Bộ lọc Gabor (41×41)\nθ = 0°, 30°, 60°, 90°, 120°, 150°",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    axes[0, 1].axis("off")

    # (c) CompCode Orientation Map
    ori_map = extractor.orientation_map(img)
    cmap_ori = plt.get_cmap("tab10", 6)
    im_ori = axes[0, 2].imshow(ori_map, cmap=cmap_ori, vmin=0, vmax=5)
    axes[0, 2].set_title(
        "(c) Bản đồ Competitive Code\nargmin phản hồi thung lũng (0–5)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    axes[0, 2].axis("off")
    cbar = fig.colorbar(im_ori, ax=axes[0, 2], ticks=range(6), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["0°", "30°", "60°", "90°", "120°", "150°"], fontsize=8)

    # (d) 10x10 Spatial Grid on ROI
    grid_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    block_px = cfg.block_pixels
    for r in range(0, 128, block_px):
        cv2.line(grid_img, (0, r), (128, r), (255, 60, 60), 1)
    for c in range(0, 128, block_px):
        cv2.line(grid_img, (c, 0), (c, 128), (255, 60, 60), 1)
    axes[0, 3].imshow(grid_img)
    axes[0, 3].set_title(
        "(d) Phân hoạch lưới không gian\n10×10 khối (14×14 px/khối)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    axes[0, 3].axis("off")

    # Feature Vector 600-D
    vec = extractor.transform(img)

    # Combine bottom row for histogram & block representation
    axes[1, 0].bar(
        range(6), vec[:6], color=plt.cm.tab10(np.arange(6)), edgecolor="black", linewidth=0.5
    )
    axes[1, 0].set_xticks(range(6))
    axes[1, 0].set_xticklabels(["0°", "30°", "60°", "90°", "120°", "150°"], fontsize=8)
    axes[1, 0].set_title(
        "(e) Histogram 6 hướng\ntại Khối [0, 0] (Chuẩn hóa L1)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    axes[1, 0].set_ylabel("Mật độ hướng")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Remove all unused bottom row subplots so they do not overlap ax_vec
    axes[1, 1].remove()
    axes[1, 2].remove()
    axes[1, 3].remove()

    # Vector 600-D full representation spanning slots 6, 7, 8
    ax_vec = fig.add_subplot(2, 4, (6, 8))
    ax_vec.plot(vec, color="#1f77b4", linewidth=0.8)
    ax_vec.fill_between(range(len(vec)), vec, color="#1f77b4", alpha=0.3)
    ax_vec.set_title(
        "(f) Vector Đặc trưng Toàn cục 600 Chiều (100 khối × 6 hướng, Chuẩn hóa L2)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax_vec.set_xlabel("Chỉ số đặc trưng (0 đến 599)")
    ax_vec.set_ylabel("Giá trị đặc trưng")
    ax_vec.set_xlim(0, 600)
    ax_vec.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Quy trình Trích xuất Đặc trưng Competitive Code (CompCode 600-D)",
        fontsize=12.5,
        fontweight="bold",
        y=0.965,
    )
    out_path = FIGDIR / "compcode_feature_extraction_pipeline.png"
    safe_savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def plot_accuracy_vs_latency():
    """Figure 3: Accuracy (Rank-1 %) vs CPU Latency (ms) Pareto Trade-off."""
    models_data = [
        {
            "name": "M3: CompCode + CRC_RLS",
            "rank1": 98.2667,
            "latency": 14.71,
            "color": "#2ca02c",
            "marker": "o",
            "pareto": True,
        },
        {
            "name": "M2: CompCode + Linear SVM",
            "rank1": 97.3667,
            "latency": 3.71,
            "color": "#1f77b4",
            "marker": "s",
            "pareto": True,
        },
        {
            "name": "M1: CompCode + 1-NN",
            "rank1": 94.4333,
            "latency": 11.00,
            "color": "#ff7f0e",
            "marker": "^",
            "pareto": False,
        },
        {
            "name": "M4-E2: ResNet18 Nearest Cosine",
            "rank1": 73.7444,
            "latency": 19.34,
            "color": "#9467bd",
            "marker": "D",
            "pareto": False,
        },
        {
            "name": "M4-E2: ResNet18 Centroid Cosine",
            "rank1": 68.8278,
            "latency": 19.34,
            "color": "#8c564b",
            "marker": "v",
            "pareto": False,
        },
        {
            "name": "M4-E1: ResNet18 Logits (Softmax)",
            "rank1": 64.3611,
            "latency": 48.72,
            "color": "#d62728",
            "marker": "X",
            "pareto": False,
        },
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Pareto frontier line between SVM and CRC
    ax.plot(
        [3.71, 14.71],
        [97.3667, 98.2667],
        linestyle="--",
        color="#2ca02c",
        linewidth=2,
        label="Biên Pareto tối ưu (Pareto Frontier)",
    )
    ax.fill_between(
        [3.71, 14.71, 55], [97.3667, 98.2667, 98.2667], [60, 60, 60], color="#2ca02c", alpha=0.06
    )

    for m in models_data:
        ax.scatter(
            m["latency"],
            m["rank1"],
            s=180,
            color=m["color"],
            marker=m["marker"],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
            label=m["name"],
        )

        # Position offset annotations cleanly without collisions
        dx, dy = 1.0, 0.8
        if "Linear SVM" in m["name"]:
            dx, dy = 1.2, -2.6
        elif "CRC_RLS" in m["name"]:
            dx, dy = -1.5, 2.2
        elif "1-NN" in m["name"]:
            dx, dy = 1.2, -2.2
        elif "Nearest Cosine" in m["name"]:
            dx, dy = 1.2, 1.6
        elif "Centroid Cosine" in m["name"]:
            dx, dy = 1.2, -3.0
        elif "Logits" in m["name"]:
            dx, dy = -15.0, 1.8

        ax.annotate(
            f"{m['name'].split(':')[1].strip()}\n({m['rank1']:.2f}%, {m['latency']:.1f} ms)",
            (m["latency"], m["rank1"]),
            xytext=(m["latency"] + dx, m["rank1"] + dy),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    ax.set_title(
        "Đánh đổi giữa Độ chính xác (Rank-1 %) và Độ trễ CPU (ms/ảnh)\nClosed-set 1:N trên 6.000 probe Tongji Session 2",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        "Độ trễ CPU đơn luồng (ms/ảnh) — Càng thấp càng nhanh", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel(
        "Tỷ lệ nhận dạng Rank-1 (%) — Càng cao càng chính xác", fontsize=10, fontweight="bold"
    )
    ax.set_xlim(0, 55)
    ax.set_ylim(60, 101)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    out_path = FIGDIR / "accuracy_vs_latency_tradeoff.png"
    safe_savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


def plot_qualitative_predictions():
    """Figure 4: Qualitative success and failure cases with probe vs gallery."""
    manifest = pd.read_csv(MANIFEST_PATH)

    # 2 Successes: Palm 1 and Palm 23
    # 2 Failures from CRC:
    # 1. Probe 00131: True Palm 14, Predicted Palm 360
    # 2. Probe 00241: True Palm 25, Predicted Palm 545
    cases = [
        {
            "type": "SUCCESS",
            "title": "Thành công #1 (Chính xác)",
            "probe_uid": "session2/00001",
            "true_id": 1,
            "pred_id": 1,
        },
        {
            "type": "SUCCESS",
            "title": "Thành công #2 (Chính xác)",
            "probe_uid": "session2/00021",
            "true_id": 3,
            "pred_id": 3,
        },
        {
            "type": "FAILURE",
            "title": "Thất bại #1 (Nhầm lẫn)",
            "probe_uid": "session2/00131",
            "true_id": 14,
            "pred_id": 360,
        },
        {
            "type": "FAILURE",
            "title": "Thất bại #2 (Nhầm lẫn)",
            "probe_uid": "session2/00241",
            "true_id": 25,
            "pred_id": 545,
        },
    ]

    fig, axes = plt.subplots(4, 3, figsize=(9.5, 11.5), dpi=300)
    plt.subplots_adjust(top=0.91, bottom=0.03, left=0.05, right=0.95, wspace=0.25, hspace=0.45)

    for row_idx, case in enumerate(cases):
        probe_path = RAW / manifest[manifest.sample_uid == case["probe_uid"]]["path"].iloc[0]
        true_s1_path = (
            RAW
            / manifest[(manifest.palm_id == case["true_id"]) & (manifest.session == "session1")][
                "path"
            ].iloc[0]
        )
        pred_s1_path = (
            RAW
            / manifest[(manifest.palm_id == case["pred_id"]) & (manifest.session == "session1")][
                "path"
            ].iloc[0]
        )

        probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
        true_img = cv2.imread(str(true_s1_path), cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(str(pred_s1_path), cv2.IMREAD_GRAYSCALE)

        # Col 0: Probe Session 2
        axes[row_idx, 0].imshow(probe_img, cmap="gray")
        axes[row_idx, 0].set_title(
            f"Probe (Session 2)\nTrue ID: {case['true_id']}", fontsize=9, fontweight="bold", pad=6
        )
        axes[row_idx, 0].axis("off")

        # Col 1: True Match (Session 1 Gallery)
        axes[row_idx, 1].imshow(true_img, cmap="gray")
        axes[row_idx, 1].set_title(
            f"Ground Truth (Session 1)\nGallery ID: {case['true_id']}",
            fontsize=9,
            color="green" if case["type"] == "SUCCESS" else "black",
            pad=6,
        )
        axes[row_idx, 1].axis("off")

        # Col 2: Predicted Top-1 (Session 1 Gallery)
        axes[row_idx, 2].imshow(pred_img, cmap="gray")
        color = "green" if case["type"] == "SUCCESS" else "red"
        label = (
            "Dự đoán ĐÚNG" if case["type"] == "SUCCESS" else f"Dự đoán SAI (ID: {case['pred_id']})"
        )
        axes[row_idx, 2].set_title(
            f"Top-1 Match (Session 1)\n{label}", fontsize=9, color=color, fontweight="bold", pad=6
        )
        axes[row_idx, 2].axis("off")

    fig.suptitle(
        "Phân tích Định tính Các Trường Hợp Nhận Dạng Đúng & Sai của Mô hình CRC_RLS\n"
        "So sánh mẫu truy vấn Session 2 với mẫu thực tế và mẫu bị nhầm lẫn trong Gallery Session 1",
        fontsize=11.5,
        fontweight="bold",
        y=0.965,
    )

    out_path = FIGDIR / "qualitative_predictions_success_failure.png"
    safe_savefig(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    plot_dataset_samples()
    plot_compcode_pipeline()
    plot_accuracy_vs_latency()
    plot_qualitative_predictions()
    print("All supplementary visuals successfully created!")
