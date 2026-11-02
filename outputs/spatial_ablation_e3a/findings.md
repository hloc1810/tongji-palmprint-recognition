# Báo cáo Thực nghiệm E3a: Phân tích Cắt giảm Thông tin Không gian (Global vs Spatial CompCode)

> **Môn học**: Nhận dạng mẫu (*Pattern Recognition*) — Đề tài 09 (Nhóm 01)  
> **Cơ sở lý thuyết**: *Pattern Classification, 2nd Edition* (Duda, Hart & Stork) — Chương Feature Extraction, Dimensionality and Invariances.  
> **Giao thức**: Đánh giá mô hình hồi quy sườn cộng tác (**CRC_RLS**, $\lambda=0.1$) qua kiểm chứng chéo 5 nếp (**5-fold Cross-Validation**) cân bằng theo Palm ID trên **Session 1** (tuyệt đối không truy cập Session 2).  
> **Trạng thái cấu hình**:
> - **Pre-registered**: Global CompCode (6-D, $1 \times 1$) và Spatial CompCode (600-D, $10 \times 10$) theo đăng ký chính thức tại `experiments/registry.yaml`.
> - **E3a Exploratory Extension**: Spatial CompCode (150-D, $5 \times 5$) nhằm khảo sát đường cong đánh đổi độ mịn không gian.

---

## 1. Bảng Kết quả Thực nghiệm Tổng hợp (Session 1 5-Fold CV)

| Biểu diễn | Trạng thái | Lưới không gian | Kích thước vector | Rank-1 Mean ± STD (%) | Rank-5 Mean ± STD (%) | Macro-F1 Mean ± STD (%) | Thời gian trích xuất (ms/ảnh) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Global CompCode** | Pre-registered | $1 \times 1$ | 6-D | **8.77% ± 0.59%** | 26.28% ± 0.32% | 3.05% ± 0.34% | 2.41 ms |
| **Spatial CompCode (5x5)** | Exploratory ext. | $5 \times 5$ | 150-D | **99.98% ± 0.04%** | 100.00% ± 0.00% | 99.98% ± 0.04% | 2.96 ms |
| **Spatial CompCode (10x10)** | Pre-registered | $10 \times 10$ | 600-D | **100.00% ± 0.00%** | 100.00% ± 0.00% | 100.00% ± 0.00% | 3.08 ms |

### Chi tiết Kết quả Từng Fold (Rank-1 %)

| Fold | Global CompCode (6-D) | Spatial CompCode 5x5 (150-D)* | Spatial CompCode 10x10 (600-D) |
| :---: | :---: | :---: | :---: |
| Fold 0 | 9.67% | 99.92% | 100.00% |
| Fold 1 | 8.33% | 100.00% | 100.00% |
| Fold 2 | 8.92% | 100.00% | 100.00% |
| Fold 3 | 8.17% | 100.00% | 100.00% |
| Fold 4 | 8.75% | 100.00% | 100.00% |

---

## 2. Phân tích Khoa học & Cơ sở Lý thuyết (Duda et al.)

### 2.1. Sự Sụp đổ Khả năng Phân biệt của Global CompCode (6-D)
- Khi nén toàn bộ ma trận định hướng kích thước $128 \times 128$ thành một histogram 6 chiều duy nhất trên toàn ảnh, độ chính xác Rank-1 trung bình giảm mạnh từ **100.00% xuống chỉ còn 8.77%**.
- **Cơ sở toán học theo Duda (Feature Extraction & Invariance)**:
  - Histogram toàn cục chỉ ghi nhận tỉ lệ phần trăm các điểm ảnh thuộc 6 hướng Gabor khác nhau ($0^\circ, 30^\circ, 60^\circ, 90^\circ, 120^\circ, 150^\circ$).
  - Trong quần thể bàn tay tự nhiên, phần lớn các nếp vân chính (đường tâm đạo, trí đạo, sinh đạo) đều có phân bố hướng tổng thể tương tự nhau giữa người này và người khác.
  - Do triệt tiêu hoàn toàn tọa độ không gian $(x, y)$, phép chiếu từ không gian ảnh $128 \times 128$ về $\mathbb{R}^6$ làm mất đi thông tin cấu trúc then chốt: nếp vân hướng $30^\circ$ nằm ở vùng gò kim tinh (dưới ngón cái) hay nằm ở vùng rìa ngoài lòng bàn tay.
  - Ma trận từ điển $\mathbf{D} \in \mathbb{R}^{6 \times 4800}$ có hạng tối đa là 6, hoàn toàn không đủ bậc tự do để phân tách 600 lớp nhãn khác nhau.

### 2.2. Vai trò của Phân hoạch Lưới Không gian (Spatial Grid Partitioning)
- Việc chia ảnh thành lưới khối không gian (Spatial Block Partitioning) biến vector đặc trưng thành biểu diễn vị trí - hướng ghép nối:
  $$\mathbf{f} = [\mathbf{h}_{1,1}^T, \mathbf{h}_{1,2}^T, \dots, \mathbf{h}_{M,M}^T]^T$$
- Ở lưới $10 \times 10$ ($M=10$, 100 khối, 600 chiều), mỗi vùng vi mô $14 \times 14$ pixel sở hữu một histogram hướng riêng biệt. Nhờ đó, mô hình lưu giữ đồng thời:
  1. Phân bố hướng cục bộ tại từng vị trí giải phẫu bàn tay.
  2. Mối quan hệ cấu trúc không gian tương đối giữa các đường chỉ tay.
- Kết quả kiểm chứng chéo đạt Rank-1 100.00% trên cả 5 folds của Session 1 khẳng định rằng **thông tin không gian cục bộ là yếu tố quyết định để tạo ra sự phân tách lớp trong bài toán nhận diện lòng bàn tay**.

### 2.3. Khảo sát Mở rộng (Exploratory Extension): Lưới Trung gian 5x5 (150-D)
- Cấu hình mở rộng $5 \times 5$ (kích thước khối 26 pixel, vector 150 chiều) đạt Rank-1 **99.98% ± 0.04%**.
- **Ý nghĩa thực tiễn**:
  - Chỉ với 150 chiều (bằng 1/4 dung lượng của lưới 10x10), mô hình đã phục hồi phần lớn năng lực phân loại, vượt xa mức 8.77% của Global CompCode.
  - Sự chênh lệch giữa 5x5 và 10x10 cho thấy độ mịn không gian ở mức 100 khối cho phép bắt trọn các đường nếp vân mảnh (*wrinkles*) tốt hơn so với lưới thô 25 khối.

---

## 3. Cảnh báo Phương pháp luận & Giới hạn Khoa học

1. **Số chiều vector khác biệt (Confounding Factor: Dimensionality)**:
   - Cần ghi nhận một cách trung thực và khách quan rằng Global CompCode có 6 chiều, trong khi Spatial CompCode có 600 chiều (gấp 100 lần).
   - Do đó, sự vượt trội của Spatial CompCode phản ánh đồng thời hai yếu tố: **sự hiện diện của tọa độ không gian** và **dung lượng biểu diễn (capacity / dimensionality)** của không gian đặc trưng. Không thể khẳng định toàn bộ chênh lệch chỉ do duy nhất thông tin không gian mà không tính đến tác động của số chiều.
2. **Không làm thay đổi kết quả chính thức**:
   - Thí nghiệm E3a được tiến hành nội bộ trên Session 1 (5-fold CV) để kiểm chứng giả thuyết kiến trúc biểu diễn đặc trưng.
   - Thí nghiệm không can thiệp, không chọn lại siêu tham số và không thay thế kết quả chính thức đã khóa của CompCode + CRC_RLS trên Session 2.

---

## 4. Đường dẫn Minh chứng & Tệp Kết quả

- Bảng dữ liệu CV chi tiết: `outputs/spatial_ablation_e3a/cv_results.csv`
- Cấu hình thực nghiệm: `outputs/spatial_ablation_e3a/experiment_config.json`
- Bằng chứng toàn vẹn & mã băm: `outputs/spatial_ablation_e3a/feature_integrity.json`
- Biểu đồ so sánh CV: `outputs/spatial_ablation_e3a/figures/spatial_vs_global_cv_comparison.png`
- Biểu đồ đánh đổi kích thước vector: `outputs/spatial_ablation_e3a/figures/spatial_granularity_tradeoff.png`
