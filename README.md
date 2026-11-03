# Tongji Contactless Palmprint Recognition

**Bổ sung ROI từ ảnh bàn tay gốc (pilot 10 Palm ID):** đã triển khai hình học Tongji 2017
và chạy so sánh trên cùng 200 ảnh, giữ nguyên CompCode + 1-NN.
Official ROI: **99/100**; Custom ROI: **98/100** probe đúng.
Xem [báo cáo ROI và các trường hợp cắt sai](reports/ROI_COMPARISON.md).
Đây là thực nghiệm 10 lớp, không so trực tiếp với benchmark 600 lớp bên dưới.

```powershell
# Chọn output mới khi chạy lại để giữ bằng chứng của lần chạy trước
.venv311/Scripts/python.exe scripts/run_roi_comparison.py --output outputs/roi_comparison_repeat
```

**M5: PCA-Gabor PalmNet + Nearest-Template Matching** — đã chạy đủ 6.000 probe:
**Rank-1 99,9833%, Rank-5 100%**, đúng 5.999/6.000 ảnh. Fit trên Session 1. Có thêm
nhánh chọn đặc trưng cho Linear SVM, CRC_RLS và đối chứng 1-NN cùng biểu diễn.
Xem [giao thức, lệnh chạy và kết quả M5](reports/PALMNET_M5.md) và
[hướng dẫn cấu trúc repository](reports/REPOSITORY_GUIDE.md).

> **Nhận diện lòng bàn tay không tiếp xúc (*Contactless Palmprint Recognition*)**  
> Đề tài 09 — Nhóm 01 | Môn Nhận dạng mẫu (*Pattern Recognition*)  
> Dataset: **Tongji Contactless Palmprint** (600 Palm ID, 12.000 ảnh ROI 128×128, 2 phiên độc lập).

---

## 1. Kết quả thực nghiệm chính (Benchmark)

### 1.1. Chuẩn đối sánh mô hình chính (Closed-set 1:N trên 6.000 probe Session 2)

| Nhóm | Pipeline | Mô hình | Rank-1 (%) | Rank-5 (%) | Macro-F1 (%) | Độ trễ CPU (ms) |
| :--- | :--- | :--- | ---:| ---:| ---:| ---:|
| **M5 bổ sung** | **M5** | **PCA-Gabor PalmNet + 1-NN** (12.288.000-D sparse) | **99.9833%** | **100%** | **99.9833%** | [Theo batch](reports/PALMNET_M5.md) |
| **Benchmark chính** | **M3** | **CompCode + CRC_RLS** ($\lambda=0.1$) | **98.27%** | **99.08%** | **97.99%** | 14.71 ms |
| | **M2** | **CompCode + Linear SVM** ($C=1$) | **97.37%** | **98.52%** | **96.82%** | **3.71 ms** |
| | **M1** | **CompCode + 1-NN** (Euclidean) | **94.43%** | **96.97%** | **93.56%** | 11.00 ms |
| | **M4-E2** | **ResNet18 Nearest Cosine** (512-D)* | 73.74% ± 0.62 | 87.50% ± 0.44 | 70.51% ± 0.81 | ~19.34 ms |
| | **M4-E1** | **ResNet18 Logits** (Softmax 600 lớp) | 64.36% ± 1.04 | 81.91% ± 0.65 | 60.79% ± 1.10 | ~48.72 ms |
| **Bóc tách đặc trưng (E3)** | **F3** | **CompCode** (600-D) + 1-NN | **94.43%** | **96.97%** | **93.56%** | 0.11 ms |
| | **F1** | **Raw Grayscale** (16.384-D) + 1-NN | 77.18% | 85.65% | 75.64% | 0.61 ms |
| | **F2** | **PCA** (256-D, fit trên Session 1) + 1-NN | 76.72% | 84.75% | 74.67% | **0.09 ms** |

*\* Ghi chú: ResNet18 Nearest Cosine là thử nghiệm hậu nghiệm (exploratory), trích xuất embedding 512-D từ mô hình phân loại để so khoảng cách Cosine; chưa phải huấn luyện metric learning chuyên biệt.*

### 1.2. Phân tích độ bền vững trước biến đổi ảnh (E6 Robustness Analysis)

Đánh giá 3 mô hình đã đóng băng trên toàn bộ 6.000 probe Session 2 dưới 5 điều kiện:
- **Độ sáng ±15**: Cả 3 mô hình đều đạt chênh lệch $\Delta = 0.00$ pp ($p = 1.0000$) nhờ tính chất triệt tiêu thành phần một chiều (Zero-DC) của bộ lọc Gabor CompCode.
- **Làm mờ Gauss 3×3**: Chênh lệch nằm trong khoảng tin cậy bao hàm 0 (KNN: +0.07 pp, SVM: +0.05 pp, CRC: -0.07 pp, $p > 0.20$), chứng minh tính ổn định của hướng nếp vân chính.
- **Góc xoay +5°**: Sụt giảm có ý nghĩa thống kê ($p = 0.0001$). CRC_RLS thể hiện độ bền cao nhất (giảm -2.52 pp xuống 95.75%), Linear SVM giảm -5.23 pp (xuống 92.13%), trong khi 1-NN sụt giảm mạnh nhất -14.12 pp (xuống 80.32%).

### 1.3. Cắt giảm thông tin không gian (E3a Global vs Spatial CompCode)

Kiểm chứng chéo 5 nếp (5-fold CV) trên Session 1 với bộ phân loại CRC_RLS ($\lambda=0.1$):
- **Global CompCode (6-D, 1×1)**: Rank-1 = **8.77% ± 0.59%** (sụp đổ khả năng phân biệt do mất hoàn toàn tọa độ không gian).
- **Spatial CompCode (150-D, 5×5)** [Exploratory extension]: Rank-1 = **99.98% ± 0.04%**.
- **Spatial CompCode (600-D, 10×10)** [Pre-registered]: Rank-1 = **100.00% ± 0.00%**.

### 1.4. So sánh thống kê cặp đôi (E3 Paired Cluster Bootstrap)

Paired Cluster Bootstrap theo Palm ID ($B=10.000$ lần, seed=42) trên 6.000 probe Session 2:
- **CompCode − Raw**: $\Delta = +17.25$ pp, 95% CI $[+15.18, +19.32]$ pp, $p < 0.0001$ (vượt trội có ý nghĩa thống kê).
- **CompCode − PCA**: $\Delta = +17.72$ pp, 95% CI $[+15.67, +19.83]$ pp, $p < 0.0001$ (vượt trội có ý nghĩa thống kê).
- **Raw − PCA**: $\Delta = +0.47$ pp, 95% CI $[-0.87, +1.83]$ pp, $p = 0.5042$ (CI bao hàm 0; không có sự khác biệt thống kê).

---

## 2. Điểm nhấn cài đặt: Tự viết (From Scratch) vs Thư viện

- **CompCode (600-D)**: Tự tạo ngân hàng 6 bộ lọc Gabor giải tích, khử DC (`kernel -= mean`), Competitive Coding ($\arg\min$ phản hồi thung lũng vân), phân hoạch lưới $10 \times 10$ và chuẩn hóa 2 cấp $L_1 \to L_2$.
- **CRC_RLS (M3)**: Tự cài đặt thuật toán Collaborative Representation Classifier từ công thức toán; hồi quy Ridge trên ma trận từ điển gallery $A \in \mathbb{R}^{600 \times 6000}$ và tính phần dư tái tạo theo từng lớp. Dùng `scipy.linalg.cho_solve` để giải nghiệm ổn định số học.
- **Nearest-Template (1-NN)**: Tự viết thuật toán tính ma trận khoảng cách Euclidean bình phương theo khối qua NumPy, không phụ thuộc `sklearn.neighbors`.
- **Cosine Embedding (512-D)**: Tự tính 600 vector trọng tâm centroid và so khớp nearest cosine theo batch.
- **Thư viện chuẩn**: `sklearn.svm.LinearSVC` (LIBLINEAR C++ cho M2) và `torchvision.models.resnet18` (PyTorch cho M4).

---

## 3. Cài đặt & Sử dụng nhanh

### 3.1. Cài đặt môi trường
```powershell
python -m pip install -e ".[test,demo,cnn]"
```

### 3.2. Chạy ứng dụng Web Demo (Streamlit)
```powershell
streamlit run app/main.py
```
> Demo gồm 4 tab: Nhận diện ảnh tương tác, X-Ray đặc trưng CompCode, Trực quan hóa không gian PCA (Eigenpalms & Reconstruction), và Dashboard tổng hợp.

### 3.3. Chạy các thực nghiệm khoa học
```powershell
# Thực nghiệm bóc tách đặc trưng (E3)
python scripts/run_feature_ablation.py

# Thực nghiệm phân tích độ bền vững trước biến đổi ảnh (E6)
python scripts/run_robustness_e6.py

# Thực nghiệm cắt giảm thông tin không gian (E3a)
python scripts/run_spatial_ablation_e3a.py

# Thực nghiệm so sánh thống kê cặp đôi bootstrap (E3)
python scripts/run_paired_bootstrap_e3.py
```

### 3.4. Chạy kiểm thử tự động
```powershell
python -m pytest -q
python scripts/run_smoke.py
```
> M5 đã qua 27 kiểm thử liên quan trong dự án và 51 kiểm thử số học upstream;
> phạm vi kiểm tra và bằng chứng được ghi trong [báo cáo M5](reports/PALMNET_M5.md).

---

## 4. Tài liệu chi tiết & Báo cáo thực nghiệm

- [Notebook 01: Phân tích Khám phá Dữ liệu (EDA)](notebooks/01_eda.ipynb)
- [Notebook 02: Phân tích Kết quả & Chẩn đoán Lỗi](notebooks/02_results_analysis.ipynb)
- [Phân tích Khám phá Dữ liệu EDA (Phân bố & Chất lượng ảnh)](reports/EDA.md)
- [Báo cáo Benchmark đầy đủ (M1–M4)](reports/TONGJI_BENCHMARK.md)
- [Báo cáo Thực nghiệm Deep CNN (ResNet18 3 Seeds)](reports/CNN_EXPERIMENTS.md)
- [Báo cáo Bóc tách Đặc trưng & Bootstrap Cặp đôi E3](outputs/feature_ablation/findings.md)
- [Báo cáo Phân tích Độ bền vững E6](outputs/robustness_e6/findings.md)
- [Báo cáo Cắt giảm Thông tin Không gian E3a](outputs/spatial_ablation_e3a/findings.md)
- [Thư viện Hình ảnh Báo cáo chất lượng cao (300 DPI)](reports/figures/)
