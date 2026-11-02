# Tongji Contactless Palmprint Recognition

> **Nhận diện lòng bàn tay không tiếp xúc (*Contactless Palmprint Recognition*)**  
> Đề tài 09 — Nhóm 01 | Môn Nhận dạng mẫu (*Pattern Recognition*)  
> Dataset: **Tongji Contactless Palmprint** (600 Palm ID, 12.000 ảnh ROI 128×128, 2 phiên độc lập).

---

## 1. Kết quả thực nghiệm chính (Benchmark)

Đánh giá closed-set 1:N trên **6.000 ảnh probe Session 2** đối chiếu với **6.000 ảnh gallery Session 1**:

| Nhóm | Pipeline | Mô hình | Rank-1 (%) | Rank-5 (%) | Macro-F1 (%) | Độ trễ CPU (ms) |
| :--- | :--- | :--- | ---:| ---:| ---:| ---:|
| **Benchmark chính** | **M3** | **CompCode + CRC_RLS** ($\lambda=0.1$) | **98.27%** | **99.08%** | **97.99%** | 14.71 ms |
| | **M2** | **CompCode + Linear SVM** ($C=1$) | **97.37%** | **98.52%** | **96.82%** | **3.71 ms** |
| | **M1** | **CompCode + 1-NN** (Euclidean) | **94.43%** | **96.97%** | **93.56%** | 11.00 ms |
| | **M4-E2** | **ResNet18 Nearest Cosine** (512-D)* | 73.74% ± 0.62 | 87.50% ± 0.44 | 70.51% ± 0.81 | ~19.34 ms |
| | **M4-E1** | **ResNet18 Logits** (Softmax 600 lớp) | 64.36% ± 1.04 | 81.91% ± 0.65 | 60.79% ± 1.10 | ~48.72 ms |
| **Bóc tách đặc trưng (E3)** | **F3** | **CompCode** (600-D) + 1-NN | **94.43%** | **96.97%** | **93.56%** | 0.11 ms |
| | **F1** | **Raw Grayscale** (16.384-D) + 1-NN | 77.18% | 85.65% | 75.64% | 0.61 ms |
| | **F2** | **PCA** (256-D, fit trên Session 1) + 1-NN | 76.72% | 84.75% | 74.67% | **0.09 ms** |

*\* Ghi chú: ResNet18 Nearest Cosine là thử nghiệm hậu nghiệm (exploratory), trích xuất embedding 512-D từ mô hình phân loại để so khoảng cách Cosine; chưa phải huấn luyện metric learning chuyên biệt.*

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

### 3.3. Chạy thực nghiệm bóc tách đặc trưng (E3)
```powershell
python scripts/run_feature_ablation.py
```

### 3.4. Chạy kiểm thử tự động
```powershell
python -m pytest -q
python scripts/run_smoke.py
```
> Toàn bộ kiểm thử tự động vượt qua (`61 passed, 1 skipped`).

---

## 4. Tài liệu chi tiết & Báo cáo thực nghiệm

- [Báo cáo Benchmark đầy đủ (M1–M4)](reports/TONGJI_BENCHMARK.md)
- [Báo cáo Thực nghiệm Deep CNN (ResNet18 3 Seeds)](reports/CNN_EXPERIMENTS.md)
- [Báo cáo Bóc tách Đặc trưng E3 (Raw vs PCA vs CompCode)](outputs/feature_ablation/findings.md)
- [Thư viện Hình ảnh Báo cáo chất lượng cao (300 DPI)](reports/figures/)
