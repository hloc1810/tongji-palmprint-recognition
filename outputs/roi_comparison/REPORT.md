# Thử nghiệm ROI Tongji 2017 trên 10 Palm ID

## Phạm vi và kết quả

Thực nghiệm thăm dò closed-set 1:N, 10 Palm ID, 200 ảnh ghép cặp.
Session 1: 100 gallery; Session 2: 100 probe. ID được chọn ngẫu nhiên không hoàn lại
với NumPy default_rng(seed=42), trước khi đánh giá, không chọn lại theo kết quả:
**52, 53, 57, 121, 258, 261, 389, 418, 459, 512**.

| Chỉ số | Official ROI | Custom ROI |
|---|---:|---:|
| Số Palm ID đã chọn | 10 | 10 |
| Số ảnh đầu vào | 200 | 200 |
| Gallery dùng được | 100 | 99 |
| Palm ID có template gallery | 10 | 10 |
| Probe cắt được | 100 | 100 |
| Tỷ lệ cắt ROI thành công theo kiểm tra tự động | Không áp dụng | 199/200 (99.50%) |
| Rank-1, mẫu số cố định 100 probe | 99/100 (99.00%) | 98/100 (98.00%) |
| Thời gian xử lý ROI trung bình/ảnh | Không áp dụng | 2.39 ms |
| ROI median / p95 | Không áp dụng | 1.90 / 2.60 ms |
| Tổng thời gian thuật toán ROI, 200 lần thử | Không áp dụng | 0.478 s |

Không so trực tiếp với **94,43% trên benchmark 600 lớp**. Cả hai Rank-1 trong bảng
đều được chạy lại trên đúng bộ 10 lớp này. Đây là pilot, không suy rộng thành kết quả 600 lớp.

"Thành công tự động" nghĩa là tìm được bộ kẽ ngón hợp lệ, tiếp tuyến, hai giao biên,
hình vuông nằm trong ảnh và ít nhất 98% ROI thuộc hand mask. Đây là tỷ lệ thuật toán
chấp nhận crop, **không phải tỷ lệ đúng giải phẫu đã gán nhãn thủ công**. SSIM với Official ROI
chỉ hỗ trợ xem ảnh, không dùng chọn mẫu, chỉnh tham số hoặc quyết định crop thành công.

Probe thất bại được tính sai trong mẫu số 100; gallery thất bại không được dùng.
Không thay bằng Official ROI, không crop tâm dự phòng, không loại probe khó khỏi mẫu số.

## Thuật toán và đối chiếu paper

Nguồn: Zhang et al., *Towards contactless palmprint recognition*, Pattern Recognition 69
(2017), 199–212, DOI 10.1016/j.patcog.2017.04.016, Section 4, Fig. 5, Table 2, trang 204–205.
[PDF tác giả](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).

1. Chuyển ảnh gốc sang grayscale OpenCV; Gaussian 5×5, sigma=1.5 để phát hiện mask.
   Ngưỡng cố định 25/255, giữ contour ngoài lớn nhất và lấp lỗ bên trong.
2. Tính góc lõm trên contour ở hai phía cách 0.0375×chiều rộng ảnh theo số điểm contour.
   Chọn ba kẽ phía trên có độ lõm lớn, khoảng cách tương đối đều và gần thẳng hàng;
   lấy kẽ ngoài cùng (trỏ–giữa và áp út–út). Tính tiếp tuyến đỡ chung phía dưới
   hai đoạn biên cục bộ bằng nghiệm số; **X1, X2 là tiếp điểm**, không chỉ là đỉnh lõm.
3. O=(X1+X2)/2; u=(X2−X1)/d; v vuông góc hướng xuống lòng bàn tay; d=|X1X2|.
   Cắt contour bằng đường O+0.60d·v+t·u để lấy O1, O2; w=|O1O2|.
4. Cạnh s=0.65w. Tâm cạnh trên T=O+0.10w·v. Bốn góc là
   T−s/2·u, T+s/2·u, T+s/2·u+s·v, T−s/2·u+s·v.
   Warp affine bilinear **ảnh grayscale gốc**, về 128×128, không blur texture đầu ra.

**Khác biệt ngay trong paper:** Table 2 bước 7 in khoảng dịch `c2|O1O2|`, trong khi
đoạn mô tả Section 4 trang 205 ghi `c3|O1O2|` và c3=0.10. Cài đặt này theo đoạn mô tả
và Hình 5: khoảng dịch đến **cạnh gần trục X nhất**, không phải tâm hình vuông.

Paper không nêu đầy đủ ngưỡng segmentation, kernel Gaussian, cách tìm kẽ và nghiệm tiếp tuyến.
Các lựa chọn kỹ thuật trên là phần tự triển khai, đã cố định khi xem Session 1 trước khi
chạy Session 2. **Không tuyên bố tái lập từng pixel hay tương đương mã nguồn tác giả.**
Giả định ảnh đúng tư thế chụp Tongji (ngón hướng lên, nền tối); không phải bộ phát hiện
bàn tay tự do ở mọi góc quay. Các phép chuẩn hóa vị trí/góc/tỷ lệ diễn ra sau phát hiện kẽ.

## CompCode và 1-NN được giữ nguyên

Dùng `CompCodeExtractor` với `configs/compcode.yaml` hiện có: 6 Gabor, 600-D spatial
histogram, chuẩn hóa L1 theo block rồi L2 toàn vector. Dùng chính `NearestTemplate` hiện có,
score lớp bằng khoảng cách Euclidean bình phương nhỏ nhất đến template của lớp,
lấy argmin trên các Palm ID tăng dần khi hòa. Không PCA, LDA, fit scaler hay mô hình mới.
CompCode của dự án vốn là bản paper-inspired, chưa xác minh parity MATLAB.

Thời gian ROI đo bằng perf_counter trên ảnh đã decode, gồm segmentation đến warp và
kiểm tra hình học, gồm cả lần bị từ chối; không gồm đọc/ghi file, CompCode, 1-NN và vẽ ảnh.
Warm-up 3 ảnh Session 1, OpenCV một luồng. Mỗi ảnh đo một lần; đây là số đo trên máy này,
có thể dao động giữa các lần chạy. Chi tiết môi trường và thời gian các khâu nằm trong summary.json.

## Bằng chứng để kiểm tra

- [Ảnh minh họa](roi_examples.png): tương đồng cao/thấp với ROI chính thức;
  thêm crop bị từ chối hoặc nhận dạng sai nếu có. **Nhận dạng sai không đồng nghĩa crop sai.**
- [Kiểm tra Session 1, ID 1–5](audit_session1_1.jpg), [ID 6–10](audit_session1_2.jpg).
- [Kiểm tra Session 2, ID 1–5](audit_session2_1.jpg), [ID 6–10](audit_session2_2.jpg).
  Mỗi ô: ảnh gốc có điểm mốc; thumbnail dưới trái Official, dưới phải Custom.
  Vàng: tiếp tuyến X1X2; xanh lam: đường đo O1O2; xanh lá: hình vuông crop.
- `samples.csv`: 200 đường dẫn và SHA256; `protocol.json`: lựa chọn/thuật toán khóa trước đánh giá.
- `roi_audit.csv`, `landmarks.json`: trạng thái, lý do lỗi, thời gian, hình học từng ảnh.
- `predictions_official.csv`, `predictions_custom.csv`: đủ 100 probe mỗi nhánh.
- `comparison.csv`, `summary.json`: kết quả máy đọc được; `custom_roi/`: ảnh 128×128.

## Chạy lại

```powershell
.venv311/Scripts/python.exe scripts/run_roi_comparison.py --output outputs/roi_comparison_repeat
```

Có thể chỉ định `--original-root`, `--official-root`, `--output` và `--seed`.
Chọn thư mục output mới mỗi lần chạy; chương trình từ chối ghi đè một thực nghiệm đã lưu.
Nếu đổi seed/thuật toán sau khi xem kết quả, đó là thực nghiệm thăm dò mới; không dùng
Session 2 để chọn biến thể tốt nhất rồi trình bày như đánh giá độc lập.
