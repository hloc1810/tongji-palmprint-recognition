# Nghiên cứu Thực nghiệm E3: Feature Representation Ablation trên Tongji Palmprint

## 1. Mục tiêu và Thiết kế Thực nghiệm

Nghiên cứu này khảo sát ảnh hưởng của **biểu diễn đặc trưng (feature representation)** đến khả năng nhận diện lòng bàn tay không tiếp xúc, cố định bộ phân loại là **1-Nearest Neighbor (k=1, Euclidean distance)**.

Thực nghiệm so sánh 3 phương pháp biểu diễn trên bộ dữ liệu Tongji Official ROI 128x128 (600 Palm ID, 12.000 ảnh):
1. **F1 — Raw Grayscale Pixels**: Vector ảnh xám 128x128 làm phẳng (16.384 chiều), chuẩn hóa L2.
2. **F2 — PCA (Principal Component Analysis)**: Giảm chiều ảnh xám xuống 256 chiều bằng Randomized SVD (seed=42). Tuân thủ nghiêm ngặt nguyên tắc **zero-leakage** (chỉ fit trên tập train của từng fold trong CV; chỉ fit trên toàn bộ Session 1 trong kiểm thử cross-session), chuẩn hóa L2 sau phép chiếu. 256 thành phần chính giải thích **93.77%** tổng phương sai của Session 1.
3. **F3 — CompCode (Competitive Code)**: Vector đặc trưng 600 chiều từ ngân hàng 6 bộ lọc Gabor 2D, trích xuất hướng đường vân nổi trội và spatial block histogram (10x10 ô), chuẩn hóa 2 cấp (L1 block -> L2 toàn cục).

> [!NOTE]
> **Quy ước đánh giá**:
> - **Giai đoạn A (Within-Session)**: 5-Fold Cross-Validation trên Session 1 (4.800 train / 1.200 val mỗi fold).
> - **Giai đoạn B (Cross-Session)**: Gallery cố định là toàn bộ Session 1 (6.000 ảnh), Query/Probe là toàn bộ Session 2 (6.000 ảnh). Thí nghiệm này là phân tích bổ sung hậu nghiệm (exploratory/post-hoc).

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm

| Biểu diễn Đặc trưng | Số chiều | Session 1: 5-Fold CV Rank-1 (%) | Session 2: Cross-Session Rank-1 (%) | Rank-1 95% CI (Session 2) | Cross-Session Rank-5 (%) | Macro-F1 (Session 2) | Thời gian khớp (ms/mẫu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1 — Raw Grayscale** | 16.384 | 99.40 ± 0.41% | **77.18%** | [74.67%, 79.53%] | 85.65% | 75.64% | 0.61 ms |
| **F2 — PCA** | 256 | 99.43 ± 0.31% | **76.72%** | [74.23%, 79.07%] | 84.75% | 74.67% | 0.09 ms |
| **F3 — CompCode** | 600 | 99.98 ± 0.04% | **94.43%** | [93.02%, 95.68%] | 96.97% | 93.56% | 0.11 ms |

---

## 3. Phân tích Kỹ thuật và Nhận xét Thực nghiệm

### 3.1. Sự khác biệt giữa kiểm định nội phiên và kiểm thử chéo phiên (Generalization Gap)
- **Trong cùng một session (5-Fold CV)**: Cả 3 phương pháp đều đạt độ chính xác Rank-1 ở mức cao (từ 99.40% đến 99.98%). Lý do là các ảnh trong cùng Session 1 được chụp liên tiếp trong cùng một phiên lấy mẫu, với điều kiện chiếu sáng, tư thế đặt tay và vị trí máy ảnh ít có sự biến động.
- **Khi kiểm thử chéo phiên (Cross-Session)**: Xuất hiện sự phân hóa rõ rệt:
  - **Raw Grayscale**: Giảm từ 99.40% xuống **77.18%** (sụt giảm 22.22 điểm phần trăm).
  - **PCA (256-D)**: Giảm từ 99.43% xuống **76.72%** (sụt giảm 22.72 điểm phần trăm).
  - **CompCode (600-D)**: Duy trì độ chính xác ổn định hơn ở mức **94.43%** (chỉ giảm 5.55 điểm phần trăm).

### 3.2. Nguyên nhân dẫn đến mức sụt giảm của Raw Pixels và PCA ở Cross-Session:
1. **Độ nhạy với nhiễu toàn cục và dịch chuyển không gian**:
   - Khoảng cách Euclidean trên vector pixel thô tính tổng sai số bậc hai theo từng pixel: d^2(x, y) = sum (x_i - y_i)^2. Khi sang Session 2 (cách Session 1 từ vài tuần đến hơn một tháng), sự thay đổi nhẹ về góc nghiêng lòng bàn tay, độ giãn da tự nhiên và cường độ ánh sáng môi trường làm toàn bộ lưới pixel bị trôi (spatial shift), khiến khoảng cách giữa hai ảnh cùng một người tăng lên đáng kể.
2. **Bản chất của PCA theo lý thuyết Duda (*Pattern Classification*)**:
   - PCA tìm các trục trực giao có phương sai lớn nhất của dữ liệu huấn luyện (Unsupervised Maximum Variance Projection). Tuy nhiên, trên ảnh xám bàn tay, các thành phần chính đầu tiên chủ yếu ghi nhận sự biến thiên cường độ ánh sáng tổng thể, vùng bóng đổ và hình thái bàn tay lớn, ít ghi nhận thông tin vi cấu trúc đường vân lòng bàn tay. Khi chiếu sang Session 2, những biến đổi về nguồn sáng làm tọa độ vector chiếu PCA bị lệch hướng.
3. **Đặc tính của CompCode**:
   - **Lọc thông dải (Bandpass Gabor)** loại bỏ thành phần một chiều (Zero-DC), giảm thiểu sự biến đổi độ sáng nền.
   - **Competitive Coding** quan tâm đến *hướng vân cục bộ* (chỉ số argmin phản hồi thực), ít phụ thuộc vào biên độ tương phản.
   - **Spatial Block Histogram (10x10)** cho phép dung sai dịch chuyển và biến dạng co giãn cục bộ nhất định.

### 3.3. Đánh giá sự đánh đổi Kích thước Vector và Tốc độ
- **PCA (256-D)**: Kích thước nhỏ nhất và tốc độ khớp nhanh nhất (0.09 ms/mẫu), nhưng độ chính xác sinh trắc học còn hạn chế trong kịch bản cross-session (76.72%).
- **Raw Grayscale (16.384-D)**: Kích thước vector lớn, tốn tài nguyên bộ nhớ và tính toán (0.61 ms/mẫu), độ chính xác ở mức trung bình (77.18%).
- **CompCode (600-D)**: Đạt điểm cân bằng giữa kích thước nén (600 chiều so với 16.384 chiều ban đầu, giảm 96,3% kích thước) và độ chính xác phân biệt sinh trắc học (94.43% Rank-1).

---

## 4. Kết luận cho Câu hỏi Nghiên cứu

> **Kết luận**:
> Trong giao thức nhận dạng cross-session trên Tongji, **CompCode cho thấy hiệu quả phân biệt cao hơn đáng kể so với Raw Grayscale và PCA** khi cùng sử dụng bộ phân loại 1-NN. Việc khai thác cấu trúc hướng vân thông qua ngân hàng bộ lọc Gabor kết hợp mã hóa thi đấu và phân đoạn không gian là yếu tố quan trọng giúp hệ thống thích ứng tốt hơn trước sự thay đổi phiên thu nhận dữ liệu.

---

## 5. So sánh Thống kê Cặp đôi (Paired Cluster Bootstrap Comparison)

> **Cơ sở lý thuyết**: *Pattern Classification, 2nd Edition* (Duda, Hart & Stork), Chương 9.6 — *Estimating and Comparing Classifiers*.  
> **Phương pháp**: Paired Cluster Bootstrap theo từng cá nhân (**Palm ID**) trên toàn bộ 6.000 ảnh probe Session 2 với $B=10.000$ lần lấy mẫu lại có hoàn lại (seed=42).  
> **Lưu ý phương pháp luận**: Do Session 2 được sử dụng làm tập kiểm thử hậu nghiệm (*exploratory/post-hoc analysis*), các khoảng tin cậy dưới đây phản ánh độ ổn định thống kê trong phân phối dữ liệu Tongji hiện có, **không được coi là bằng chứng xác nhận độc lập (out-of-domain confirmation)** cho các hệ thống camera thực tế khác.

### 5.1. Bảng Kết quả So sánh Cặp đôi

| Cặp so sánh ($M_A - M_B$) | Rank-1 $M_A$ (%) | Rank-1 $M_B$ (%) | Chênh lệch quan sát (pp) | Khoảng tin cậy 95% Bootstrap (pp) | Độ lệch chuẩn Bootstrap (pp) | p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CompCode - Raw Grayscale** | **94.43%** | 77.18% | **+17.25 pp** | **[+15.18, +19.32] pp** | 1.06 pp | < 0.0001 |
| **CompCode - PCA (256-D)** | **94.43%** | 76.72% | **+17.72 pp** | **[+15.67, +19.83] pp** | 1.07 pp | < 0.0001 |
| **Raw Grayscale - PCA (256-D)** | 77.18% | 76.72% | **+0.47 pp** | **[-0.87, +1.83] pp** | 0.68 pp | 0.5042 |

### 5.2. Diễn giải Khoa học theo Duda et al.

1. **CompCode vượt trội có ý nghĩa thống kê so với Raw Pixels và PCA**:
   - Khoảng tin cậy 95% của chênh lệch giữa CompCode và Raw Grayscale là $[+15.18, +19.32]$ điểm phần trăm, nằm hoàn toàn bên phải điểm 0 ($p < 0.0001$).
   - Tương tự, CompCode vượt trội PCA với khoảng tin cậy $[+15.67, +19.83]$ điểm phần trăm ($p < 0.0001$).
   - Điều này xác nhận rằng ưu thế của CompCode trước sự thay đổi phiên chụp (*cross-session covariate shift*) là hoàn toàn thực chất và ổn định khi lấy mẫu lại theo từng cá nhân, không phải là biến động ngẫu nhiên do một nhóm nhỏ Palm ID quyết định.

2. **Raw Pixels và PCA không có sự khác biệt mang ý nghĩa thống kê**:
   - Chênh lệch quan sát giữa Raw Grayscale (77.18%) và PCA (76.72%) chỉ là $+0.47$ điểm phần trăm.
   - Khoảng tin cậy Bootstrap 95% là $[-0.87, +1.83]$ điểm phần trăm, bao hàm giá trị 0 ($p = 0.5042$).
   - **Phân tích nguyên nhân**:
     - Phép chiếu tuyến tính không giám sát PCA giữ lại 256 thành phần chính giải thích $93.77\%$ tổng phương sai của Session 1.
     - Tuy nhiên, trong nhận diện ảnh xám lòng bàn tay, các hướng biến thiên phương sai lớn nhất trong Session 1 chủ yếu gắn liền với mức độ chiếu sáng tổng thể và độ phản xạ nền. Khi kiểm thử sang Session 2 với ánh sáng thay đổi, không gian con PCA không tách biệt được nếp vân đặc thù tốt hơn không gian pixel gốc.
     - Dù vậy, PCA mang lại lợi thế vượt trội về kích thước vector (giảm từ 16.384 chiều xuống 256 chiều, tiết kiệm 98,4% bộ nhớ) và tốc độ khớp nhanh gấp gần 7 lần (0.09 ms so với 0.61 ms), trong khi duy trì hiệu năng tương đương về mặt thống kê với Raw Pixels.

### 5.3. Minh chứng Thực nghiệm Đi kèm
- Tệp dữ liệu thống kê cặp đôi: `outputs/feature_ablation/paired_bootstrap_results.csv`
- Cấu hình và mã băm toàn vẹn: `outputs/feature_ablation/bootstrap_config.json`
- Biểu đồ phân phối hiệu số bootstrap: `outputs/feature_ablation/figures/paired_feature_differences.png`
