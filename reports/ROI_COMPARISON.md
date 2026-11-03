# Tự triển khai ROI Tongji 2017: pilot 10 Palm ID

Đã hoàn thành bốn phần: thuật toán ROI từ ảnh gốc, bộ mẫu 200 ảnh ghép cặp,
CompCode + 1-NN, bảng kết quả và minh họa crop đúng/sai.

## Kết quả trên cùng bộ mẫu

Chọn ngẫu nhiên 10 Palm ID bằng NumPy default_rng(seed=42):
**52, 53, 57, 121, 258, 261, 389, 418, 459, 512**.
Mỗi ID có 10 ảnh Session 1 làm gallery và 10 ảnh Session 2 làm probe.
Ghép ảnh Original và Official ROI theo session/tên ảnh; không chọn lại ID theo kết quả.

| Chỉ số | Official ROI | Custom ROI |
|---|---:|---:|
| Số Palm ID | 10 | 10 |
| Số ảnh đầu vào | 200 | 200 |
| Tỷ lệ cắt ROI thành công theo kiểm tra tự động | Không áp dụng | **199/200 = 99,5%** |
| Gallery dùng được | 100 | 99 |
| Probe dùng được | 100 | 100 |
| Rank-1 | **99/100 = 99%** | **98/100 = 98%** |
| Thời gian ROI trung bình | Không áp dụng | **2,39 ms/ảnh** |

[Bảng, thời gian và giao thức đầy đủ](../outputs/roi_comparison/REPORT.md) ·
[CSV kết quả](../outputs/roi_comparison/comparison.csv) ·
[Môi trường và thời gian đo](../outputs/roi_comparison/summary.json).

Thời gian đo riêng thuật toán trên ảnh đã decode: median 1,90 ms, p95 2,60 ms;
tổng 200 lần thử 0,478 giây. Không gồm đọc/ghi file, CompCode hay 1-NN.
Các lần chạy lặp cùng mẫu cho cùng 199 crop và cùng dự đoán; thời gian có dao động theo tải máy.

**99,5% là tỷ lệ crop được thuật toán chấp nhận, không phải tỷ lệ đúng giải phẫu.**
Kiểm tra overlay đã phát hiện ít nhất bốn crop được chấp nhận nhưng chọn sai bộ kẽ:
`session1/02578`, `session2/02574`, `session2/02575`, `session2/02578`.
Điểm lõm ở cạnh ngón bị chọn thay cho một kẽ thật, làm trục xoay và ROI lệch.
Các crop này vẫn được giữ trong thực nghiệm để phản ánh hành vi thực tế của thuật toán.
Một ảnh khác, `session1/02575`, bị từ chối do tỷ lệ nền 2,19%, vượt giới hạn 2%.
Không thay ảnh này bằng crop tâm hoặc Official ROI.

Chưa có bộ nhãn landmark/crop độc lập cho toàn bộ 200 ảnh; vì vậy chưa công bố một
tỷ lệ đúng giải phẫu. Đánh giá trực quan từng ví dụ và tỷ lệ chấp nhận tự động là hai thông tin khác nhau.

Custom ROI thấp hơn Official ROI **1 điểm phần trăm**, tương ứng một probe.
Đây là pilot nhỏ; chưa đủ để kết luận hai phương pháp tương đương hoặc khác biệt có ý nghĩa thống kê.
Không sử dụng **94,43% của benchmark 600 lớp** để so trực tiếp với kết quả 10 lớp này.

## Đối chiếu paper và giới hạn cài đặt

Nguồn: Zhang et al., *Towards contactless palmprint recognition: A novel device, a new
benchmark, and a collaborative representation based identification approach*, Pattern
Recognition 69 (2017), 199–212, DOI 10.1016/j.patcog.2017.04.016.
[PDF trên trang tác giả](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).

Hình học theo Section 4, Fig. 5, trang 204–205:

1. Gaussian và phân đoạn nền tối để tạo hand mask.
2. Tìm biên hai kẽ trỏ–giữa và áp út–út, tính tiếp tuyến chung với tiếp điểm X1, X2.
3. Lập hệ trục tại O=(X1+X2)/2. Đường đo chiều rộng song song X1X2,
   dịch **0,60×|X1X2|** vào lòng bàn tay; giao biên tại O1, O2.
4. ROI đối xứng qua trục Y, cạnh **0,65×|O1O2|**; khoảng cách từ trục X
   đến **cạnh trên** là **0,10×|O1O2|**. Warp affine về grayscale **128×128**.

Table 2 bước 7 in khoảng dịch là c2, nhưng đoạn văn Section 4 dùng c3=0.10.
Cài đặt theo đoạn văn và Fig. 5; không dùng 0.10 làm độ dịch của tâm ROI.

Paper không nêu đầy đủ ngưỡng segmentation và thuật toán số cho tìm kẽ/tiếp tuyến.
Bản triển khai dùng Gaussian 5×5, sigma=1.5; ngưỡng 25; contour lớn nhất; độ lõm contour
để chọn bộ ba kẽ; tiếp tuyến đỡ chung tính bằng nghiệm số. Các lựa chọn được cố định
trên Session 1 trước khi đánh giá Session 2. Không điều chỉnh để sửa các lỗi probe đã thấy.
Đây là **bản tự triển khai theo hình học paper**, chưa xác minh tương đương mã tác giả.
Giả định ảnh Tongji nền tối, ngón hướng lên; chưa hỗ trợ tư thế bàn tay bất kỳ.

CompCode 600-D và `NearestTemplate` Euclidean của dự án được tái sử dụng nguyên vẹn.
Không thêm mô hình, PCA, LDA hay huấn luyện mạng. Mỗi nhánh dùng ROI tương ứng cho cả
gallery và probe; không trộn Official gallery với Custom probe. Mọi probe cắt thất bại
sẽ được tính sai trong mẫu số cố định 100; không loại ảnh khó để tăng Rank-1.

## Mã nguồn và bằng chứng

- [Thuật toán ROI](../src/palmprint/data/tongji_roi.py).
- [Chương trình chạy thực nghiệm](../scripts/run_roi_comparison.py).
- [Kiểm thử hình học, biến đổi và xử lý thất bại](../tests/test_tongji_roi.py).
- [Minh họa tự động: tương đồng cao/thấp, crop bị từ chối, nhận dạng sai](../outputs/roi_comparison/roi_examples.png).
- [Một crop hợp lý và bốn ví dụ chọn sai kẽ đã kiểm tra trực quan](../outputs/roi_comparison/visual_review_examples.png),
  [nhãn ví dụ](../outputs/roi_comparison/visual_review.csv),
  [script vẽ lại minh họa có nhãn](../scripts/render_roi_visual_review.py).
- [Danh sách mẫu, đường dẫn và SHA256](../outputs/roi_comparison/samples.csv).
- [Tham số và SHA256 mã nguồn khóa trước đánh giá](../outputs/roi_comparison/protocol.json).
- [Nhật ký trạng thái crop](../outputs/roi_comparison/roi_audit.csv),
  [tọa độ điểm mốc](../outputs/roi_comparison/landmarks.json).
- [100 dự đoán Official](../outputs/roi_comparison/predictions_official.csv),
  [100 dự đoán Custom](../outputs/roi_comparison/predictions_custom.csv).

Bốn contact sheet trong báo cáo tự sinh hiển thị đầy đủ 200 ảnh, không chỉ ảnh chọn đẹp.
Nhận dạng sai không tự động có nghĩa crop sai: ví dụ `session2/02609` nhận dạng sai dù
ROI nhìn vẫn nằm hợp lý trong lòng bàn tay. Ngược lại, crop sai bộ kẽ vẫn có thể nhận dạng đúng.

Đã qua **31 kiểm thử liên quan** (13 kiểm thử mới cho ROI, cùng các kiểm thử đọc ảnh,
CompCode, mô hình, dataset và liên kết tài liệu); Ruff không báo lỗi. Đã xác minh SHA256
mã nguồn khóa, cả 400 file đầu vào, đủ 200 mẫu ghép cặp, 199 ROI grayscale 128×128 và
dự đoán lặp lại giống lần đầu: [biên bản kiểm tra](../outputs/roi_comparison/validation.json).

## Chạy lại và dùng riêng thuật toán

```powershell
# Dùng môi trường sẵn có; output mới để không ghi đè bằng chứng cũ
.venv311/Scripts/python.exe scripts/run_roi_comparison.py --output outputs/roi_comparison_repeat

# Kiểm thử phần bổ sung
.venv311/Scripts/python.exe -m pytest tests/test_tongji_roi.py -q -p no:cacheprovider
```

Có thể đặt `--original-root`, `--official-root`, `--seed`, `--output`.
Mặc định đọc `data/Original Images` và `data/raw` của workspace hiện tại, không phụ thuộc
đường dẫn ổ đĩa cũ trong metadata benchmark. Chương trình không sửa metadata, mô hình,
cache hay kết quả benchmark chính.

```python
import cv2
from palmprint.data.tongji_roi import extract_roi, ROIExtractionError

image = cv2.imread("data/Original Images/session1/00001.tiff")
try:
    result = extract_roi(image)
    cv2.imwrite("roi.png", result.roi)  # uint8, grayscale 128×128
except ROIExtractionError as error:
    print("Không cắt được ROI:", error)
```

Có thể trình bày đóng góp là: “Tự triển khai tiền xử lý ROI theo hình học Tongji 2017,
đánh giá ghép cặp 10 Palm ID với CompCode + 1-NN, đo tốc độ và phân tích lỗi landmark”.
Không trình bày như một thuật toán ROI mới hoặc kết quả tái lập chính xác mã gốc.
