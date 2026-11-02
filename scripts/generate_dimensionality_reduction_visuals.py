"""Generate high-resolution (300 DPI) dimensionality reduction visualizations for Tongji.

Generates:
1. pca_eigenpalms_visualization.png: Mean Palm and Top-11 Eigenpalms (Principal Components)
2. pca_reconstruction_progression.png: Palm reconstruction under k in {5, 20, 50, 100, 256, 500}
3. pca_session_shift_scatter.png: 2D projection (PC1 vs PC2) showing the cross-session domain shift
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from palmprint.data.preparation import load_manifest


def normalize_display(img: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] for clean visual display."""
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax - vmin < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def generate_eigenpalms_plot(pca: PCA, save_paths: list[Path]) -> None:
    """Plot Mean Palm and Top 11 Eigenpalms in a 2x6 grid."""
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), dpi=300)
    axes = axes.ravel()

    # 1. Mean Palm
    mean_img = pca.mean_.reshape(128, 128)
    axes[0].imshow(normalize_display(mean_img), cmap="gray")
    axes[0].set_title("Mean Palm (μ)\n[Lòng bàn tay TB]", fontsize=9, fontweight="bold", pad=6)
    axes[0].axis("off")

    # 2. Top 11 Eigenpalms
    var_ratios = pca.explained_variance_ratio_ * 100.0
    for idx in range(11):
        ax = axes[idx + 1]
        comp_img = pca.components_[idx].reshape(128, 128)
        # Academic grayscale visualization of eigen-vectors (standard publication style)
        ax.imshow(normalize_display(comp_img), cmap="gray")
        ax.set_title(
            f"Eigenpalm #{idx + 1}\n({var_ratios[idx]:.1f}% Var)",
            fontsize=9,
            fontweight="bold",
            pad=6,
        )
        ax.axis("off")

    fig.suptitle(
        "Không gian các thành phần chính (Eigenpalms) từ Session 1 (PCA trên 6.000 mẫu)\n"
        "Các vector riêng đầu tiên chủ yếu mô tả biến thiên cường độ sáng và hình thái diện rộng của lòng bàn tay.",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Eigenpalms figure to: {save_paths[0]}")


def generate_reconstruction_progression(
    pca: PCA,
    x_s1: np.ndarray,
    x_s2: np.ndarray,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    save_paths: list[Path],
) -> None:
    """Plot palm reconstruction progression over increasing number of components k."""
    k_values = [5, 20, 50, 100, 256, 500]

    # Select sample 0 from Session 1 and sample 0 from Session 2 (Palm ID 1)
    sample_s1 = x_s1[0]
    sample_s2 = x_s2[0]

    fig, axes = plt.subplots(2, len(k_values) + 1, figsize=(16, 5.2), dpi=300)

    for row_idx, (sample, title_prefix) in enumerate(
        [(sample_s1, "Session 1 (Palm ID 1)"), (sample_s2, "Session 2 (Palm ID 1)")]
    ):
        # Original Image
        orig_img = sample.reshape(128, 128)
        axes[row_idx, 0].imshow(normalize_display(orig_img), cmap="gray")
        axes[row_idx, 0].set_title(
            f"Gốc ({title_prefix})\n16.384 chiều",
            fontsize=8.5,
            fontweight="bold",
            pad=6,
        )
        axes[row_idx, 0].axis("off")

        # Project sample onto all components
        z = pca.transform(sample[None, :])[0]

        # Reconstruct under each k
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx + 1]
            rec = pca.mean_ + z[:k] @ pca.components_[:k]
            mse = float(np.mean((sample - rec) ** 2))
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            compression = (1.0 - (k / 16384.0)) * 100.0

            rec_img = rec.reshape(128, 128)
            ax.imshow(normalize_display(rec_img), cmap="gray")
            is_e3_dim = " (E3 Config)" if k == 256 else ""
            ax.set_title(
                f"k = {k}{is_e3_dim}\nPSNR: {psnr:.1f} dB | -{compression:.1f}%",
                fontsize=8,
                fontweight="bold" if k == 256 else "normal",
                color="#2b5c8f" if k == 256 else "#2d3748",
                pad=6,
            )
            ax.axis("off")

    fig.suptitle(
        "Tái tạo ảnh lòng bàn tay qua các mức số chiều PCA (k từ 5 đến 500)\n"
        "Ở k=256 (cấu hình E3), ảnh tái tạo thể hiện hình thái và phân bố sáng tổng thể; các nếp vân tần số cao bị giảm độ nét.",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Reconstruction figure to: {save_paths[0]}")


def generate_session_shift_scatter(
    pca: PCA,
    x_s1: np.ndarray,
    x_s2: np.ndarray,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    save_paths: list[Path],
) -> None:
    """Plot 2D projection (PC1 vs PC2) showing the session shift / domain drift."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    # Choose 5 representative Palm IDs with elegant academic palette (ColorBrewer/Dark2)
    target_ids = [1, 2, 3, 4, 5]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#4a5568"]

    # Transform all samples of target IDs
    z_s1 = pca.transform(x_s1)
    z_s2 = pca.transform(x_s2)

    labels_s1 = df_s1.palm_id.to_numpy()
    labels_s2 = df_s2.palm_id.to_numpy()

    for idx, pid in enumerate(target_ids):
        mask1 = labels_s1 == pid
        mask2 = labels_s2 == pid

        pts1 = z_s1[mask1, :2]
        pts2 = z_s2[mask2, :2]

        # Session 1: Circles
        ax.scatter(
            pts1[:, 0],
            pts1[:, 1],
            color=colors[idx],
            marker="o",
            s=45,
            alpha=0.8,
            label=f"Palm ID {pid} (Session 1)",
            edgecolors="black",
            linewidth=0.6,
        )

        # Session 2: Triangles
        ax.scatter(
            pts2[:, 0],
            pts2[:, 1],
            color=colors[idx],
            marker="^",
            s=55,
            alpha=0.8,
            label=f"Palm ID {pid} (Session 2)",
            edgecolors="black",
            linewidth=0.6,
        )

        # Draw drift arrow between Session 1 centroid and Session 2 centroid
        c1 = pts1.mean(axis=0)
        c2 = pts2.mean(axis=0)
        ax.annotate(
            "",
            xy=(c2[0], c2[1]),
            xytext=(c1[0], c1[1]),
            arrowprops=dict(
                arrowstyle="->",
                color=colors[idx],
                lw=1.8,
                linestyle="--",
                alpha=0.9,
            ),
        )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% Variance)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Trực quan hóa dịch chuyển phiên (Session Shift) trên không gian 2D PCA (PC1 vs PC2)\n"
        "Mũi tên nét đứt thể hiện xu hướng dịch chuyển tọa độ trung bình của từng Palm ID giữa Session 1 và Session 2.",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    for p in save_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Session Shift figure to: {save_paths[0]}")


def main() -> None:
    ws = Path(".").resolve()
    print("=== Generating Dimensionality Reduction Visuals (PCA / Eigenpalms) ===")

    # Load manifest and raw cached data
    df, _ = load_manifest(ws)
    df_s1 = df[df.session == "session1"].reset_index(drop=True)
    df_s2 = df[df.session == "session2"].reset_index(drop=True)

    cache_s1 = ws / "outputs/feature_ablation/cache/raw_session1.npz"
    cache_s2 = ws / "outputs/feature_ablation/cache/raw_session2.npz"

    if not cache_s1.is_file() or not cache_s2.is_file():
        raise FileNotFoundError("Run scripts/run_feature_ablation.py first to extract raw caches.")

    print("Loading cached raw pixel arrays...")
    x_s1 = np.load(cache_s1)["X"]
    x_s2 = np.load(cache_s2)["X"]
    print(f"Loaded Session 1: {x_s1.shape}, Session 2: {x_s2.shape}")

    print("Fitting PCA (256 components, randomized SVD, seed=42)...")
    pca = PCA(n_components=256, svd_solver="randomized", random_state=42)
    pca.fit(x_s1)

    # Destination paths
    out_dir1 = ws / "outputs/feature_ablation/figures"
    out_dir2 = ws / "reports/figures"

    # 1. Eigenpalms plot
    print("Generating Eigenpalms visualization...")
    generate_eigenpalms_plot(
        pca,
        [
            out_dir1 / "pca_eigenpalms_visualization.png",
            out_dir2 / "pca_eigenpalms_visualization.png",
        ],
    )

    # 2. Reconstruction progression
    print("Generating Palm Reconstruction progression...")
    generate_reconstruction_progression(
        pca,
        x_s1,
        x_s2,
        df_s1,
        df_s2,
        [
            out_dir1 / "pca_reconstruction_progression.png",
            out_dir2 / "pca_reconstruction_progression.png",
        ],
    )

    # 3. Session shift scatter
    print("Generating Session Shift scatter plot...")
    generate_session_shift_scatter(
        pca,
        x_s1,
        x_s2,
        df_s1,
        df_s2,
        [
            out_dir1 / "pca_session_shift_scatter.png",
            out_dir2 / "pca_session_shift_scatter.png",
        ],
    )

    print("\nAll dimensionality reduction visuals generated successfully at 300 DPI!")


if __name__ == "__main__":
    main()
