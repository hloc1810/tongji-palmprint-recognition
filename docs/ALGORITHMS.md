# Đặc trưng và mô hình nhận dạng

## ROI → Competitive Code → Histogram

- ROI được cung cấp sẵn từ Tongji: 128×128 pixel. Ảnh grayscale chuyển sang float32 0–1.
- Tạo 6 bộ lọc Gabor định hướng `θ_j = πj/6`, j=0..5. Tham số mặc định σx=4,85; σy=9,31; f=0,071. Kernel hữu hạn 41×41; chuẩn zero-DC hữu hạn và chuẩn L2, phép tích chập `reflect101`. Gabor real response được chọn theo `argmin` cho từng pixel.
- Bản đồ mã `{0..5}` được chia thành các block kích thước **14 pixel**, với block biên chưa đủ vẫn được giữ. 128×128 → ceil(128/14)=10 block/chiều → 100 block, mỗi block 6 histogram bin, tổng **600 chiều**. Histogram mỗi block chuẩn L1; vector nối chuẩn L2. Global ablation có 6 chiều.
- HOG ablation: 128×128, 9 hướng, cell8, block2, L2-Hys, vector 8100 chiều chuẩn L2.
- Các tham số lấy cảm hứng từ paper Tongji; **không khẳng định kernel/biên/phase/normalization tương đương mã MATLAB tác giả**, cần đối chiếu tham chiếu trước khi dùng từ "reproduction".

## M1 — 1-NN nearest-template

Với truy vấn y và gallery của lớp c:

`score_c(y) = min_{j : label_j=c} || y - v_j ||²₂`

Sắp tăng 600 scores, Top-1 là class tốt nhất. Tính squared distance (không lấy căn) không đổi thứ tự so Euclidean.

## M2 — Linear SVM

`LinearSVC` với squared-hinge loss, multi-class One-vs-Rest, C thử nghiệm trên session1 CV. Không dùng `predict_proba` vì margin không phải xác suất. Chuyển `decision_function` thành negative margin để quy ước score càng thấp càng tốt, không sửa ranking.

Với mỗi lớp c, đặt `t_ic=+1` khi mẫu i thuộc c, ngược lại `-1`. Bản chạy dùng `penalty=l2`, `fit_intercept=True`, `intercept_scaling=1`, `tol=1e-4`, `dual=False`, `max_iter=30000`. Liblinear thêm một synthetic feature bằng 1 và phạt cả hệ số intercept. Objective mỗi bài toán OvR là:

`min_{w_c,b_c} 1/2 (||w_c||²₂ + b_c²) + C Σ_i max(0, 1 - t_ic(w_cᵀx_i+b_c))²`

`score_c(x) = -(w_cᵀx+b_c)`.

Đây là squared-hinge SVM. Nghiệm analytic `W=(XᵀX+λI)⁻¹XᵀT` tối thiểu hóa `||XW-T||²_F+λ||W||²_F`: đó là multi-output ridge regression/classification không intercept, không tương đương SVM này hoặc LS-SVM chuẩn. Workspace hiện tại không có bộ Scratch SVM đó và không cung cấp bằng chứng cho accuracy 98,05% từ bản cũ.

Nguồn: [LinearSVC, đặc biệt loss và intercept_scaling](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

## M3 — CRC_RLS

Cho gallery dictionary `A ∈ ℝ^{m×n}`, mỗi cột L2-unit, truy vấn `y ∈ ℝ^m`:

`x* = argmin_x ||y - Ax||²₂ + λ||x||²₂`

`x* = Aᵀ (AAᵀ+λI)⁻¹ y`  (giải hệ tuyến tính, **không nghịch đảo tường minh**)

`score_c = ||y - A_c x_c||₂`

Tên biến thể dùng trong báo cáo: **CRC ridge với residual Euclidean không chuẩn hóa**. Không chia score cho `||x_c||₂`. Eq. (6) của [paper Tongji, §5](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf) dùng `||y-A_c x_c||²₂`; lấy căn giữ ranking nhưng thay giá trị score. Biến thể chia cho norm hệ số lớp là một quy tắc quyết định khác, có thể đổi nhãn dự đoán, chỉ được khảo sát như thí nghiệm riêng có đăng ký trước.

Triển khai factor hóa Cholesky `AAᵀ+λI` một lần khi fit. Để tính residual nhanh dùng:

`r_c² = ||y||² + x_cᵀ(A_cᵀA_c)x_c - 2x_cᵀA_cᵀy`

Không tạo 600 vector reconstruction chiều m cho mỗi truy vấn. Unit test đối chiếu lời giải ridge trực tiếp trên ma trận nhỏ.

## M4 — ResNet18

Pretrained ResNet18 ImageNet, ROI grayscale lặp thành 3 kênh, Resize 224×224, normalization ImageNet. Thay head classification bằng số lớp thực tế (600 khi chạy strict). Train head trước, thử unfreeze layer4; chọn lịch epoch bằng validation fold0 của session1; khởi tạo lại rồi train trên toàn session1 trước test. Train augmentation chỉ trong session1; không flip ngang tùy ý. Mỗi seed là một run riêng, không chọn seed bằng test.

## Ranh giới lý thuyết

- k-NN: Duda, Hart, Stork, *Pattern Classification*, Ch. 4.
- SVM: Ch. 5; Neural Networks: Ch. 6; Classifier evaluation: Ch. 9.
- CompCode, CRC_RLS và kiến trúc ResNet18 cần dẫn các bài báo riêng.
- So sánh k-NN/SVM/CRC trên cùng feature chỉ cô lập thuật toán phân loại; CNN dùng learned features nên chỉ so hiệu quả **pipeline**.
