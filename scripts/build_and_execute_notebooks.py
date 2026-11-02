"""Script to build and execute clean, production-grade Jupyter Notebooks for EDA and Results/Error Analysis."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_eda_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()

    # Cell 1: Header
    nb.cells.append(
        new_markdown_cell(
            """# Tongji Contactless Palmprint — Phân tích Khám phá Dữ liệu (EDA)

> **Môn học**: Nhận dạng mẫu (*Pattern Recognition*) — Đề tài 09 (Nhóm 01)  
> **Bộ dữ liệu**: Tongji Contactless Palmprint Dataset (600 Palm ID, 12.000 ảnh ROI mức xám $128 \\times 128$)  
> **Mục tiêu Notebook**:
> 1. Khám phá cấu trúc, quy mô và tính cân bằng của tập dữ liệu giữa 2 phiên (Session 1 và Session 2).
> 2. Phân tích thống kê định lượng các thuộc tính chất lượng ảnh (Cường độ sáng, Độ tương phản, Độ sắc nét Laplacian).
> 3. Trực quan hóa mẫu ảnh thực tế và các dạng biến thiên nội lớp (Intra-class variations: ánh sáng, góc xoay, co giãn da).
> 4. Khám phá không gian đặc trưng và giảm chiều PCA (Eigenpalms, Scree plot và Session Shift 2D)."""
        )
    )

    # Cell 2: Setup
    nb.cells.append(
        new_code_cell(
            """from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Cấu hình đường dẫn và font hiển thị học thuật
WS = Path("..").resolve()
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

manifest_path = WS / "data/metadata/manifest.csv"
df = pd.read_csv(manifest_path)
print(f"Đã nạp manifest thành công: {len(df):,} mẫu ảnh ROI từ {df.palm_id.nunique()} Palm ID.")"""
        )
    )

    # Cell 3: Structure Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 1. Cấu trúc & Phân bố số lượng mẫu (Class & Session Balance)
Kiểm tra số lượng mẫu ảnh trên từng phiên và tính cân bằng lớp giữa 600 bàn tay độc lập."""
        )
    )

    # Cell 4: Class counts
    nb.cells.append(
        new_code_cell(
            """# Thống kê tổng hợp số lượng mẫu theo phiên
summary_df = (
    df.groupby("session")
    .agg(
        So_Palm_ID=("palm_id", "nunique"),
        Tong_so_anh=("sample_uid", "count"),
        Anh_moi_Palm_min=("palm_id", lambda x: x.value_counts().min()),
        Anh_moi_Palm_max=("palm_id", lambda x: x.value_counts().max()),
    )
    .reset_index()
)
display(summary_df)

# Biểu đồ kiểm chứng tính cân bằng lớp (10 ảnh/phiên trên toàn bộ 600 Palm ID)
pivot = df.groupby(["session", "palm_id"]).size().unstack(0, fill_value=0)

fig, ax = plt.subplots(figsize=(9, 3.5), dpi=150)
ax.plot(pivot.index, pivot["session1"], label="Session 1 (Gallery)", color="#2b5c8f", lw=1.5)
ax.plot(pivot.index, pivot["session2"], label="Session 2 (Probe)", color="#d95f02", lw=1.2, ls="--")
ax.set_xlabel("Chỉ số Palm ID (1 đến 600)", fontsize=10, fontweight="bold")
ax.set_ylabel("Số lượng ảnh / lớp", fontsize=10, fontweight="bold")
ax.set_title("Phân bố số lượng mẫu: Cố định đúng 10 ảnh/phiên trên 600 Palm ID", fontsize=11, fontweight="bold")
ax.set_ylim(0, 15)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(frameon=True)
plt.tight_layout()
plt.show()"""
        )
    )

    # Cell 5: Quality Metrics Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 2. Phân tích Thuộc tính Chất lượng Ảnh (Image Quality Analysis)
Đo lường 3 chỉ số cấp độ pixel trên 12.000 ảnh:
- **Brightness**: Cường độ sáng trung bình pixel.
- **Contrast**: Độ lệch chuẩn mức xám (độ tương phản).
- **Blur Proxy**: Biến sai toán tử Laplace $\\text{Var}(\\nabla^2 I)$ đo độ sắc nét nếp vân."""
        )
    )

    # Cell 6: Quality Stats & Histograms
    nb.cells.append(
        new_code_cell(
            """# Bảng thống kê định lượng chất lượng ảnh Session 1 vs Session 2
quality_stats = (
    df.groupby("session")[["brightness", "contrast", "blur_proxy"]]
    .agg(["mean", "std", "min", "max"])
    .round(2)
)
display(quality_stats)

# Trực quan hóa phân bố 3 chỉ số chất lượng ảnh (1x3 panel)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
metrics = [
    ("brightness", "Cường độ sáng (Mean Intensity)", [0, 255]),
    ("contrast", "Độ tương phản (Pixel Std)", None),
    ("blur_proxy", "Độ sắc nét (Laplacian Var)", None),
]
colors = {"session1": "#2b5c8f", "session2": "#d95f02"}
labels = {"session1": "Session 1 (Gallery)", "session2": "Session 2 (Probe)"}

for idx, (col, title, xlim) in enumerate(metrics):
    ax = axes[idx]
    for s_name in ["session1", "session2"]:
        data = df[df.session == s_name][col]
        m = data.mean()
        ax.hist(data, bins=32, alpha=0.6, color=colors[s_name], edgecolor="black", lw=0.4, label=f"{labels[s_name]} (TB: {m:.1f})")
        ax.axvline(m, color=colors[s_name], linestyle="--", lw=1.5)
    ax.set_xlabel(title, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Số lượng ảnh ROI", fontsize=9.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5)
    if xlim:
        ax.set_xlim(xlim)

fig.suptitle("Phân bố Thuộc tính Chất lượng Ảnh: Session 1 (Gallery) vs Session 2 (Probe)", fontsize=12, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()"""
        )
    )

    # Cell 7: Intra-Class Variations Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 3. Trực quan hóa Biến thiên Nội lớp (Visual Intra-Class Variations)
Minh họa sự khác biệt trực quan giữa ảnh chụp Session 1 (Gallery) và Session 2 (Probe) của cùng một người sau khoảng thời gian vài tuần đến một tháng."""
        )
    )

    # Cell 8: Image Samples
    nb.cells.append(
        new_code_cell(
            """# Trực quan hóa mẫu ảnh thực tế của 4 Palm ID tiêu biểu
palms = [1, 25, 126, 403]
raw_dir = WS / "data/raw"

fig, axes = plt.subplots(len(palms), 4, figsize=(10, 9.5), dpi=150)

for row_idx, pid in enumerate(palms):
    s1_paths = df[(df.palm_id == pid) & (df.session == "session1")]["path"].head(2).tolist()
    s2_paths = df[(df.palm_id == pid) & (df.session == "session2")]["path"].head(2).tolist()
    items = [
        (s1_paths[0], f"Palm {pid} · S1 (#1)\\n(Gallery)"),
        (s1_paths[1], f"Palm {pid} · S1 (#2)\\n(Gallery)"),
        (s2_paths[0], f"Palm {pid} · S2 (#1)\\n(Probe)"),
        (s2_paths[1], f"Palm {pid} · S2 (#2)\\n(Probe)"),
    ]
    for col_idx, (rel_p, title) in enumerate(items):
        ax = axes[row_idx, col_idx]
        img = cv2.imread(str(raw_dir / rel_p), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8.5, fontweight="bold" if "S2" in title else "normal", pad=4)
        ax.axis("off")

fig.suptitle("Minh họa Biến thiên Chiếu sáng, Góc xoay và Độ co giãn Nếp vân giữa 2 Phiên", fontsize=11.5, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()"""
        )
    )

    # Cell 9: Feature Space EDA Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 4. Khám phá Không gian Đặc trưng & Giảm chiều PCA (Feature-Space EDA)
Sử dụng Principal Component Analysis (PCA 256 chiều, fit trên Session 1) để phân tích cấu trúc phương sai và hiện tượng trôi dạt miền dữ liệu (*Domain Shift*)."""
        )
    )

    # Cell 10: PCA Scree Plot
    nb.cells.append(
        new_code_cell(
            """# Nạp cache raw pixel đã trích xuất từ 6.000 ảnh Session 1
cache_s1 = WS / "outputs/feature_ablation/cache/raw_session1.npz"
cache_s2 = WS / "outputs/feature_ablation/cache/raw_session2.npz"

x_s1 = np.load(cache_s1)["X"]
x_s2 = np.load(cache_s2)["X"]

pca = PCA(n_components=256, svd_solver="randomized", random_state=42)
pca.fit(x_s1)

exp_var = pca.explained_variance_ratio_ * 100.0
cum_var = np.cumsum(exp_var)
print(f"PCA 256 chiều giải thích tổng cộng {cum_var[-1]:.2f}% phương sai dữ liệu Session 1.")

# Scree plot phân bố phương sai
fig, ax1 = plt.subplots(figsize=(8, 3.8), dpi=150)
comp_idx = np.arange(1, len(exp_var) + 1)
ax1.plot(comp_idx, exp_var, color="#2b5c8f", lw=1.5, label="Phương sai riêng phần (%)")
ax1.set_xlabel("Chỉ số thành phần chính (PC)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Phương sai riêng phần (%)", color="#2b5c8f", fontsize=10, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(comp_idx, cum_var, color="#d95f02", lw=1.8, ls="--", label="Phương sai tích lũy (%)")
ax2.set_ylabel("Phương sai tích lũy (%)", color="#d95f02", fontsize=10, fontweight="bold")
ax2.set_ylim(0, 105)

plt.title(f"PCA Scree Plot: 256 chiều giữ lại {cum_var[-1]:.2f}% tổng phương sai", fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.show()"""
        )
    )

    # Cell 11: Eigenpalms
    nb.cells.append(
        new_code_cell(
            """# Trực quan hóa Mean Palm và Top 5 Eigenpalms
fig, axes = plt.subplots(1, 6, figsize=(14, 2.6), dpi=150)

# Mean Palm
axes[0].imshow(pca.mean_.reshape(128, 128), cmap="gray")
axes[0].set_title("Mean Palm (μ)", fontsize=9, fontweight="bold", pad=4)
axes[0].axis("off")

for i in range(5):
    comp = pca.components_[i].reshape(128, 128)
    comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-12)
    axes[i + 1].imshow(comp_norm, cmap="gray")
    axes[i + 1].set_title(f"PC #{i+1}\\n({exp_var[i]:.1f}% Var)", fontsize=8.5, fontweight="bold", pad=4)
    axes[i + 1].axis("off")

fig.suptitle("Không gian Vector riêng (Eigenpalms): Mô tả cường độ sáng diện rộng và hình thái lòng bàn tay", fontsize=11, fontweight="bold", y=1.04)
plt.tight_layout()
plt.show()"""
        )
    )

    # Cell 12: Session Shift 2D
    nb.cells.append(
        new_code_cell(
            """# Trực quan hóa dịch chuyển phiên (Session Shift) trên 2D (PC1 vs PC2) của 5 Palm ID
target_ids = [1, 2, 3, 4, 5]
colors_dark2 = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3", "#4a5568"]

z_s1 = pca.transform(x_s1)
z_s2 = pca.transform(x_s2)
df_s1 = df[df.session == "session1"].reset_index(drop=True)
df_s2 = df[df.session == "session2"].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

for idx, pid in enumerate(target_ids):
    pts1 = z_s1[df_s1.palm_id == pid, :2]
    pts2 = z_s2[df_s2.palm_id == pid, :2]
    
    ax.scatter(pts1[:, 0], pts1[:, 1], color=colors_dark2[idx], marker="o", s=40, alpha=0.8, edgecolors="black", lw=0.5, label=f"Palm {pid} (S1)")
    ax.scatter(pts2[:, 0], pts2[:, 1], color=colors_dark2[idx], marker="^", s=50, alpha=0.8, edgecolors="black", lw=0.5, label=f"Palm {pid} (S2)")
    
    # Mũi tên dịch chuyển trọng tâm
    c1, c2 = pts1.mean(axis=0), pts2.mean(axis=0)
    ax.annotate("", xy=c2, xytext=c1, arrowprops=dict(arrowstyle="->", color=colors_dark2[idx], lw=1.6, ls="--"))

ax.set_xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% Var)", fontsize=9.5, fontweight="bold")
ax.set_ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% Var)", fontsize=9.5, fontweight="bold")
ax.set_title("Trực quan hóa Session Shift trên không gian 2D PCA (Mũi tên nét đứt thể hiện độ trôi dạt trọng tâm)", fontsize=10.5, fontweight="bold", pad=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, frameon=True)
plt.tight_layout()
plt.show()"""
        )
    )

    return nb


def build_results_analysis_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()

    # Cell 1: Header
    nb.cells.append(
        new_markdown_cell(
            """# Tongji Contactless Palmprint — Phân tích Kết quả & Chẩn đoán Lỗi (Error Diagnostics)

> **Môn học**: Nhận dạng mẫu (*Pattern Recognition*) — Đề tài 09 (Nhóm 01)  
> **Giao thức**: Closed-set 1:N đối chiếu 6.000 probe Session 2 với 6.000 gallery Session 1.  
> **Mục tiêu Notebook**:
> 1. Tổng hợp và so sánh trực quan hiệu năng nhận dạng giữa các mô hình kinh điển (M1–M3), Deep Learning (M4) và Bóc tách đặc trưng (E3).
> 2. Phân tích đánh đổi Pareto giữa Độ chính xác (Rank-1 %) và Độ trễ CPU (ms/ảnh).
> 3. Chẩn đoán nguyên nhân lỗi: Thống kê số ca sai, xác định các cặp Palm ID bị nhầm lẫn nhiều nhất (*most confused pairs*).
> 4. Phân tích tương quan giữa lỗi nhận diện với thuộc tính ảnh (Độ sáng, Độ tương phản, Độ sắc nét) và minh họa ca thất bại trực quan."""
        )
    )

    # Cell 2: Setup
    nb.cells.append(
        new_code_cell(
            """from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WS = Path("..").resolve()
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

print("Đã khởi tạo môi trường phân tích kết quả.")"""
        )
    )

    # Cell 3: Benchmark Summary Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 1. Bảng Tổng hợp Hiệu năng Benchmark & Ablation
So sánh toàn diện Rank-1, Rank-5, Macro-F1 và Độ trễ trên 6.000 ảnh kiểm thử Session 2."""
        )
    )

    # Cell 4: Benchmark Table
    nb.cells.append(
        new_code_cell(
            """# Bảng kết quả tổng hợp chính thức
clean_table = pd.DataFrame([
    {"Mô hình": "M3: CompCode + CRC_RLS (λ=0.1)", "Rank-1 (%)": 98.27, "Rank-5 (%)": 99.08, "Macro-F1 (%)": 97.99, "Độ trễ CPU (ms)": 14.71},
    {"Mô hình": "M2: CompCode + Linear SVM (C=1)", "Rank-1 (%)": 97.37, "Rank-5 (%)": 98.52, "Macro-F1 (%)": 96.82, "Độ trễ CPU (ms)": 3.71},
    {"Mô hình": "M1: CompCode + 1-NN (Euclidean)", "Rank-1 (%)": 94.43, "Rank-5 (%)": 96.97, "Macro-F1 (%)": 93.56, "Độ trễ CPU (ms)": 11.00},
    {"Mô hình": "M4-E2: ResNet18 Nearest Cosine (512-D)*", "Rank-1 (%)": 73.74, "Rank-5 (%)": 87.50, "Macro-F1 (%)": 70.51, "Độ trễ CPU (ms)": 19.34},
    {"Mô hình": "M4-E1: ResNet18 Logits (Softmax 600 lớp)", "Rank-1 (%)": 64.36, "Rank-5 (%)": 81.91, "Macro-F1 (%)": 60.79, "Độ trễ CPU (ms)": 48.72},
    {"Mô hình": "E3: Raw Grayscale (16.384-D) + 1-NN", "Rank-1 (%)": 77.18, "Rank-5 (%)": 85.65, "Macro-F1 (%)": 75.64, "Độ trễ CPU (ms)": 0.61},
    {"Mô hình": "E3: PCA (256-D) + 1-NN", "Rank-1 (%)": 76.72, "Rank-5 (%)": 84.75, "Macro-F1 (%)": 74.67, "Độ trễ CPU (ms)": 0.09},
])
display(clean_table.set_index("Mô hình"))"""
        )
    )

    # Cell 5: Pareto Trade-off Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 2. Đánh đổi Pareto: Độ chính xác vs Độ trễ CPU
Phân tích mô hình tối ưu giữa chất lượng nhận diện sinh trắc và thời gian đáp ứng thời gian thực."""
        )
    )

    # Cell 6: Pareto Plot
    nb.cells.append(
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)

# Vẽ biên Pareto giữa SVM và CRC
ax.plot([3.71, 14.71], [97.37, 98.27], ls="--", color="#2ca02c", lw=2, label="Biên Pareto tối ưu (Pareto Frontier)")
ax.fill_between([3.71, 14.71, 52], [97.37, 98.27, 98.27], [60, 60, 60], color="#2ca02c", alpha=0.06)

points = [
    ("M3: CRC_RLS", 14.71, 98.27, "#2ca02c", "o", (-1.5, 2.0)),
    ("M2: Linear SVM", 3.71, 97.37, "#1f77b4", "s", (1.2, -2.6)),
    ("M1: 1-NN", 11.00, 94.43, "#ff7f0e", "^", (1.2, -2.2)),
    ("M4-E2: Nearest Cosine", 19.34, 73.74, "#9467bd", "D", (1.2, 1.6)),
    ("M4-E1: Softmax Logits", 48.72, 64.36, "#d62728", "X", (-15.0, 1.8)),
]

for name, lat, acc, col, marker, (dx, dy) in points:
    ax.scatter(lat, acc, s=120, color=col, marker=marker, edgecolors="black", lw=1, zorder=5)
    ax.annotate(f"{name}\\n({acc:.1f}%, {lat:.1f} ms)", (lat, acc), xytext=(lat + dx, acc + dy),
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

ax.set_xlabel("Độ trễ CPU đơn luồng (ms/ảnh) — Càng thấp càng nhanh", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Độ chính xác Rank-1 (%) — Càng cao càng tốt", fontsize=9.5, fontweight="bold")
ax.set_title("Đánh đổi giữa Độ chính xác Rank-1 (%) và Độ trễ CPU (ms/ảnh)", fontsize=11, fontweight="bold", pad=10)
ax.set_xlim(0, 52)
ax.set_ylim(60, 102)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True, fontsize=8.5)
plt.tight_layout()
plt.show()"""
        )
    )

    # Cell 7: Error Analysis Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 3. Chẩn đoán Lỗi Nhận dạng (Error Analysis & Diagnostics)
Khảo sát chi tiết 104 ca dự đoán sai của mô hình tốt nhất (M3: CRC_RLS) trên 6.000 ảnh kiểm thử Session 2."""
        )
    )

    # Cell 8: Error Stats
    nb.cells.append(
        new_code_cell(
            """# Tải bảng dự đoán chi tiết của CRC_RLS
crc_pred_path = WS / "outputs/runs/crc/predictions.csv"
manifest_path = WS / "data/metadata/manifest.csv"

df_pred = pd.read_csv(crc_pred_path)
df_manifest = pd.read_csv(manifest_path)

# Ghép với metadata chất lượng ảnh
df_merged = df_pred.merge(df_manifest[["sample_uid", "brightness", "contrast", "blur_proxy"]], on="sample_uid")

n_total = len(df_merged)
n_correct = int(df_merged.correct.sum())
n_wrong = n_total - n_correct

print(f"Tổng số mẫu kiểm thử Session 2: {n_total:,}")
print(f"Dự đoán chính xác:               {n_correct:,} ({n_correct/n_total*100:.2f}%)")
print(f"Dự đoán nhầm lẫn:                {n_wrong:,} ({n_wrong/n_total*100:.2f}%)")

# Top 10 cặp Palm ID bị nhầm lẫn nhiều nhất
confused_pairs = (
    df_merged[df_merged.correct == 0]
    .groupby(["true_palm_id", "pred_palm_id"])
    .size()
    .reset_index(name="so_lan_nham")
    .sort_values(by="so_lan_nham", ascending=False)
    .head(10)
)
print("\\nTop 10 cặp Palm ID bị nhầm lẫn nhiều nhất:")
display(confused_pairs.reset_index(drop=True))"""
        )
    )

    # Cell 9: Error vs Quality Correlation
    nb.cells.append(
        new_code_cell(
            """# So sánh chất lượng ảnh giữa mẫu Đúng vs mẫu Sai
comp_quality = (
    df_merged.groupby("correct")[["brightness", "contrast", "blur_proxy"]]
    .agg(["mean", "std"])
    .round(2)
)
comp_quality.index = ["Dự đoán Sai (104 ảnh)", "Dự đoán Đúng (5.896 ảnh)"]
display(comp_quality)

# Trực quan hóa tương quan phân bố độ sắc nét (Laplacian Var) giữa ca đúng và ca sai
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)

# Histogram độ sắc nét
ax1.hist(df_merged[df_merged.correct == 1]["blur_proxy"], bins=30, alpha=0.6, color="#2ca02c", density=True, label="Đúng (Mean: 27.8)")
ax1.hist(df_merged[df_merged.correct == 0]["blur_proxy"], bins=30, alpha=0.6, color="#d62728", density=True, label="Sai (Mean: 26.2)")
ax1.set_xlabel("Độ sắc nét nếp vân (Laplacian Var)", fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Mật độ xác suất", fontsize=9.5, fontweight="bold")
ax1.set_title("Phân bố Độ sắc nét giữa ca Đúng và Sai", fontsize=10.5, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True)

# Boxplot so sánh độ tương phản
data_box = [df_merged[df_merged.correct == 1]["contrast"], df_merged[df_merged.correct == 0]["contrast"]]
ax2.boxplot(data_box, tick_labels=["Đúng", "Sai"], patch_artist=True,
            boxprops=dict(facecolor="#e2e8f0"), medianprops=dict(color="#1a365d", lw=1.5))
ax2.set_ylabel("Độ tương phản (Pixel Std)", fontsize=9.5, fontweight="bold")
ax2.set_title("Độ tương phản giữa ca Đúng và Sai", fontsize=10.5, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()"""
        )
    )

    # Cell 10: Qualitative Cases Markdown
    nb.cells.append(
        new_markdown_cell(
            """## 4. Trực quan hóa Định tính: Ca Thành công vs Ca Thất bại
Đối chiếu trực tiếp ảnh truy vấn Session 2 với ảnh đăng ký đúng (Ground Truth) và ảnh bị nhầm lẫn (False Match) trong Session 1."""
        )
    )

    # Cell 11: Qualitative Inspection Plot
    nb.cells.append(
        new_code_cell(
            """# Lấy 1 ca đúng và 2 ca sai tiêu biểu để minh họa
raw_dir = WS / "data/raw"

cases = [
    {"type": "SUCCESS", "title": "Thành công #1 (Chính xác)", "probe_uid": "session2/00001", "true_id": 1, "pred_id": 1},
    {"type": "FAILURE", "title": "Thất bại #1 (Nhầm lẫn)", "probe_uid": "session2/00131", "true_id": 14, "pred_id": 360},
    {"type": "FAILURE", "title": "Thất bại #2 (Nhầm lẫn)", "probe_uid": "session2/00241", "true_id": 25, "pred_id": 545},
]

fig, axes = plt.subplots(len(cases), 3, figsize=(9.5, 8.2), dpi=150)

for row_idx, case in enumerate(cases):
    probe_p = raw_dir / df_manifest[df_manifest.sample_uid == case["probe_uid"]]["path"].iloc[0]
    true_p = raw_dir / df_manifest[(df_manifest.palm_id == case["true_id"]) & (df_manifest.session == "session1")]["path"].iloc[0]
    pred_p = raw_dir / df_manifest[(df_manifest.palm_id == case["pred_id"]) & (df_manifest.session == "session1")]["path"].iloc[0]

    axes[row_idx, 0].imshow(cv2.imread(str(probe_p), cv2.IMREAD_GRAYSCALE), cmap="gray")
    axes[row_idx, 0].set_title(f"Probe (Session 2)\\nTrue ID: {case['true_id']}", fontsize=8.5, fontweight="bold", pad=4)
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(cv2.imread(str(true_p), cv2.IMREAD_GRAYSCALE), cmap="gray")
    axes[row_idx, 1].set_title(f"Ground Truth (Session 1)\\nGallery ID: {case['true_id']}", fontsize=8.5, pad=4)
    axes[row_idx, 1].axis("off")

    axes[row_idx, 2].imshow(cv2.imread(str(pred_p), cv2.IMREAD_GRAYSCALE), cmap="gray")
    status_label = "Dự đoán ĐÚNG" if case["type"] == "SUCCESS" else f"Dự đoán SAI (ID: {case['pred_id']})"
    status_col = "green" if case["type"] == "SUCCESS" else "red"
    axes[row_idx, 2].set_title(f"Top-1 Match (Session 1)\\n{status_label}", fontsize=8.5, color=status_col, fontweight="bold", pad=4)
    axes[row_idx, 2].axis("off")

fig.suptitle("Đối chiếu Ca Nhận dạng Thành công & Ca Thất bại của CRC_RLS", fontsize=11.5, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()"""
        )
    )

    return nb


def main():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building & Executing 01_eda.ipynb ===")
    nb_eda = build_eda_notebook()
    client1 = NotebookClient(
        nb_eda, timeout=120, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}
    )
    client1.execute()
    out_path1 = notebooks_dir / "01_eda.ipynb"
    with open(out_path1, "w", encoding="utf-8") as f:
        nbformat.write(nb_eda, f)
    print(f"Saved executed notebook to: {out_path1}")

    print("=== Building & Executing 02_results_analysis.ipynb ===")
    nb_results = build_results_analysis_notebook()
    client2 = NotebookClient(
        nb_results,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": "notebooks"}},
    )
    client2.execute()
    out_path2 = notebooks_dir / "02_results_analysis.ipynb"
    with open(out_path2, "w", encoding="utf-8") as f:
        nbformat.write(nb_results, f)
    print(f"Saved executed notebook to: {out_path2}")

    print("All notebooks built and executed successfully!")


if __name__ == "__main__":
    main()
