# Phân tích Khám phá Dữ liệu (Exploratory Data Analysis — EDA)

> **Bộ dữ liệu**: Tongji Contactless Palmprint Dataset  
> **Quy mô**: 600 Palm ID · 12.000 ảnh ROI mức xám $128 \times 128$  
> **Thư viện đầu ra**: [`reports/figures/`](figures/) và [`reports/tables/`](tables/)

---

## 1. Tổng quan Cấu trúc & Phân bố Mẫu

Bộ dữ liệu Tongji được thu nhận qua thiết bị chụp lòng bàn tay không tiếp xúc chuyên dụng từ **300 tình nguyện viên** (mỗi người cung cấp 2 bàn tay trái và phải $\rightarrow$ **600 Palm ID độc lập**):

| Thuộc tính phân bố | Session 1 (Gallery) | Session 2 (Probe) | Toàn bộ Dataset |
| :--- | :---: | :---: | :---: |
| **Mục đích sử dụng** | Huấn luyện & Đăng ký (Enrollment) | Kiểm thử liên phiên (Cross-Session Test) | Closed-set 1:N Benchmark |
| **Số lượng Palm ID** | 600 lớp | 600 lớp | 600 lớp |
| **Số ảnh / Palm ID** | Cố định đúng **10 ảnh** | Cố định đúng **10 ảnh** | Cố định đúng **20 ảnh** |
| **Tổng số ảnh ROI** | **6.000 ảnh** | **6.000 ảnh** | **12.000 ảnh** |
| **Tính cân bằng lớp** | **Cân bằng tuyệt đối (100%)** | **Cân bằng tuyệt đối (100%)** | **Cân bằng tuyệt đối (100%)** |
| **Kích thước ảnh ROI** | $128 \times 128$ pixel | $128 \times 128$ pixel | $128 \times 128$ pixel |
| **Định dạng & Kênh** | 8-bit Grayscale (0–255) | 8-bit Grayscale (0–255) | 8-bit Grayscale (0–255) |

> [!NOTE]
> Phân bố số lượng mẫu giữa các lớp là **cân bằng hoàn hảo** (balanced dataset, không bị mất cân bằng lớp - imbalanced data), giúp loại bỏ hoàn toàn hiện tượng thiên lệch tần suất lớp (*frequency bias*) khi tính toán Macro-F1 hay Rank-k.

---

## 2. Thống kê Định lượng Chất lượng Ảnh (Image Quality Metrics)

Chất lượng ảnh ROI được đo lường tự động qua 3 chỉ số cấp độ pixel trên toàn bộ 12.000 ảnh:

1. **Cường độ sáng trung bình (Brightness)**: $\mu = \frac{1}{N}\sum_{i=1}^N I_i$
2. **Độ tương phản (Contrast)**: Độ lệch chuẩn mức xám $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (I_i - \mu)^2}$
3. **Độ sắc nét nếp vân (Sharpness / Blur Proxy)**: Biến sai toán tử Laplace $\text{Var}(\nabla^2 I)$, phản ánh mật độ và độ tương phản của biên cạnh vi mô.

### Bảng đối chiếu chất lượng Session 1 vs Session 2:

| Chỉ số chất lượng | Phiên thu nhận | Trung bình (Mean) | Độ lệch chuẩn (Std) | Giá trị nhỏ nhất (Min) | Giá trị lớn nhất (Max) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cường độ sáng** *(Brightness)* | **Session 1** | **131.50** | 26.35 | 67.72 | 222.47 |
| | **Session 2** | **132.59** | 22.40 | 64.70 | 208.32 |
| **Độ tương phản** *(Contrast)* | **Session 1** | **17.65** | 5.08 | 6.20 | 39.26 |
| | **Session 2** | **17.80** | 4.41 | 7.82 | 35.08 |
| **Độ sắc nét** *(Laplacian Var)* | **Session 1** | **27.58** | 14.05 | 4.22 | 101.64 |
| | **Session 2** | **27.79** | 12.94 | 4.97 | 116.03 |

*(Trích xuất từ file [`reports/tables/eda_quality_summary.csv`](tables/eda_quality_summary.csv))*.

---

## 3. Nhận xét & Ý nghĩa Sinh trắc học

### 3.1. Tính ổn định vĩ mô giữa 2 phiên thu nhận
* **Phân bố toàn cục tương đồng cao**: Giá trị trung bình của độ sáng ($131.50$ vs $132.59$), độ tương phản ($17.65$ vs $17.80$) và độ sắc nét ($27.58$ vs $27.79$) giữa hai phiên chụp hầu như không có sự khác biệt có ý nghĩa thống kê ở cấp độ toàn bộ dataset.
* Thiết bị thu nhận của Đại học Đồng Tế có nguồn sáng LED định chuẩn tương đối đồng nhất, không xuất hiện hiện tượng trôi dạt độ sáng tổng thể (*global illumination shift*).

### 3.2. Hiện tượng dịch chuyển miền cục bộ (Local Domain / Session Drift)
Mặc dù thống kê toàn cục ổn định, việc phân tích chi tiết từng đối tượng (*intra-class variation*) chỉ ra những thách thức sinh trắc học then chốt:
1. **Dịch chuyển không gian (Spatial Translation & Rotation)**: Do thu nhận không tiếp xúc, người dùng đặt tay tự do trong không gian chụp, gây ra độ xoay nhỏ ($\pm 5^\circ - 10^\circ$) và xê dịch tâm bàn tay giữa 2 phiên.
2. **Biến dạng đàn hồi da (Skin Elasticity Distortion)**: Lực mở lòng bàn tay khác nhau giữa 2 phiên làm co giãn nhẹ các nếp vân chính (*principal lines*) và nếp nhăn phụ (*wrinkles*).
3. **Ý nghĩa thực nghiệm**:
   - Khoảng cách Euclidean trên pixel thô (Raw Grayscale) và vector PCA toàn cục bị nhạy cảm mạnh với dịch chuyển không gian, dẫn đến sụt giảm Rank-1 từ **99.4% (nội phiên CV)** xuống **76.7% - 77.2% (liên phiên Session 2)**.
   - Ngược lại, bộ lọc Gabor định hướng của CompCode (khử DC, thi đấu hướng argmin, phân hoạch khối $10 \times 10$) cung cấp khả năng dung sai mạnh mẽ với các dịch chuyển cục bộ này, duy trì độ chính xác cao **94.43%**.

---

## 4. Danh mục Biểu đồ EDA trong Báo cáo

| Tên file hình ảnh | Vị trí lưu trữ | Nội dung minh họa |
| :--- | :--- | :--- |
| `eda_quality_distributions.png` | [`reports/figures/`](figures/eda_quality_distributions.png) | **Bảng tổng hợp phân bố 3 chỉ số (1x3 panel, 300 DPI)**: Độ sáng, độ tương phản và độ sắc nét Session 1 vs Session 2 kèm vạch trung bình. |
| `dataset_samples_session1_vs_session2.png` | [`reports/figures/`](figures/dataset_samples_session1_vs_session2.png) | **Mẫu ảnh thực tế**: Minh họa biến thiên nội lớp (Gallery Session 1 vs Probe Session 2) của 4 Palm ID tiêu biểu. |
| `eda_class_counts.png` | [`reports/figures/`](figures/eda_class_counts.png) | **Cân bằng số lượng mẫu**: Chứng minh tính cân bằng 100% của 600 lớp đối tượng. |
| `pca_session_shift_scatter.png` | [`reports/figures/`](figures/pca_session_shift_scatter.png) | **Dịch chuyển miền dữ liệu 2D**: Minh họa hiện tượng Session Shift giữa 2 phiên chụp trên không gian thành phần chính. |
