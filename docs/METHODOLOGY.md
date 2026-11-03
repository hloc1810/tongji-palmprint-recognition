# Phương pháp và giao thức đánh giá

## 1. Phát biểu bài toán

Cho gallery G đã biết nhãn Palm ID và probe q, hệ thống trả về các Palm ID
được xếp hạng. Đây là **closed-set identification 1:N**: nhãn probe thuộc tập
lớp gallery trong thiết kế ban đầu. Không đánh giá người lạ hoặc xác minh 1:1.

Dataset có 600 Palm ID, hai session, 10 ảnh/ID/session. Ảnh thứ n trong mỗi
session có nhãn `floor((n−1)/10)+1`, n=1…6.000. UID là `session/stem` vì cùng
tên ở hai phiên không phải cùng ảnh. Hai tay của một người là hai lớp riêng.

## 2. Hai đường tiền xử lý

### Official ROI cho benchmark chính

Decoder kiểm tra uint8 và kích thước 128×128; ảnh màu được đổi grayscale bằng
OpenCV. Không resize ảnh toàn bàn tay thành Official ROI. Các phương pháp
được so sánh trên cùng danh sách ảnh và nhãn trong manifest.

### Custom ROI cho pilot

Hình học dựa trên Section 4, Fig. 5 của [Zhang et al. (2017)](https://doi.org/10.1016/j.patcog.2017.04.016):
tạo mask, tìm tiếp điểm X1/X2 tại kẽ trỏ–giữa và áp út–út, dựng hệ trục tại
trung điểm O, đo bề rộng bàn tay và crop theo tỷ lệ.

Với d=|X1X2|, w=|O1O2|, u dọc X1X2 và v hướng vào lòng bàn tay:

- Đường đo bề rộng: `O + 0.60 d v + t u`.
- Cạnh ROI: `s = 0.65 w`.
- Tâm cạnh gần trục X: `T = O + 0.10 w v`.
- Các góc: `T ± s/2 u` và `T ± s/2 u + s v`.

Khoảng 0.10w áp dụng cho cạnh gần nhất, không phải tâm ROI. Đoạn văn paper
dùng c3 cho khoảng dịch, khác c2 in ở Table 2 bước 7; cài đặt theo đoạn văn
và hình minh họa. Gaussian chỉ dùng phát hiện mask; ROI lấy từ grayscale gốc
bằng warp affine bilinear về 128×128.

Ngưỡng segmentation và cách tìm kẽ là lựa chọn kỹ thuật tự triển khai,
không được paper mô tả đầy đủ. Pilot dùng 10 ID chọn trước bằng seed 42,
100 gallery và 100 probe. Probe crop thất bại vẫn tính sai trong mẫu số 100;
gallery crop thất bại không được dùng. Không thay bằng Official ROI dự phòng.
Tỷ lệ vượt kiểm tra hình học khác tỷ lệ đúng giải phẫu có nhãn độc lập.
[Báo cáo ROI](../reports/ROI_COMPARISON.md) ghi đầy đủ lỗi và phạm vi timing.

## 3. Biểu diễn và nhận dạng

| Pipeline | Biểu diễn | Chọn/học cấu hình | Quy tắc nhận dạng |
|---|---|---|---|
| M1 | CompCode 600-D | Extractor cố định | 1-NN Euclidean |
| M2 | Cùng CompCode | C của LinearSVC qua CV Session 1 | Margin OvR |
| M3 | Cùng CompCode | Lambda CRC_RLS qua CV Session 1 | Phần dư nhỏ nhất |
| M4 | ResNet18 logits/512-D | Epoch trên fold0 Session 1, refit Session 1 | Logits; cosine hậu nghiệm |
| M5 | PalmNet sparse 12.288.000-D | Bộ lọc không giám sát trên Session 1 | Nearest template |

CompCode dùng sáu Gabor thực, chọn hướng đáp ứng nhỏ nhất, histogram theo
block 14 pixel và giữ block biên: 10×10×6=600 chiều. Chuẩn hóa L1 từng block,
rồi L2 toàn vector. Đây là bản paper-inspired, chưa xác minh parity MATLAB.

1-NN chấm điểm lớp `D(q,c)=min_g ||f(q)−f(g)||²`, với g thuộc lớp c.
CRC_RLS giải ridge trên toàn gallery, dùng phần dư Euclidean không chia cho
chuẩn hệ số lớp. `ranking()` xếp điểm nhỏ trước, ưu tiên Palm ID tăng dần khi
hòa. Top-5 là năm lớp khác nhau, không phải năm template của cùng một lớp.

M5 dùng Python port PCA→Gabor tại commit ghi trong [báo cáo M5](../reports/PALMNET_M5.md),
không khẳng định tái lập tất cả biến thể PalmNet. Nhánh chọn 1.024 cột bằng
phương sai Session 1 có đối chứng 1-NN trên cùng biểu diễn để tách ảnh hưởng
của classifier khỏi ảnh hưởng chọn đặc trưng.

## 4. Chọn cấu hình và kiểm soát rò rỉ

M1–M3 dùng 5-fold CV trong Session 1: 4.800 train/1.200 validation mỗi fold.
Chọn mean CV Rank-1 tốt nhất; hòa chọn tham số nhỏ hơn. Fit final trên đủ
6.000 ảnh Session 1. Model/protocol khóa trước đánh giá tương ứng Session 2.

CNN dùng một fold Session 1 để chọn lịch epoch, khác CV năm fold của classical.
Seed 17, 29, 43 được báo cáo cùng nhau, không chọn seed theo test. PCA hoặc bộ
chọn đặc trưng trong validation phải fit trên train fold, không fit trên cả
validation/test.

Nghiên cứu triển khai tuần tự; embedding cosine là hậu nghiệm sau khi đã xem
Session 2. Khóa từng run không biến toàn bộ nghiên cứu thành một test hoàn toàn
chưa từng được quan sát. Lịch sử này được giữ trong [báo cáo CNN](../reports/CNN_EXPERIMENTS.md).

## 5. Chỉ số và đơn vị

- **Rank-k:** tỷ lệ probe có nhãn thật trong k Palm ID đứng đầu.
- **Macro-F1:** trung bình F1 của các lớp, dùng dự đoán Top-1.
- **CMC:** Rank-k theo toàn bộ thang k; không phải ROC xác minh.
- **CI bootstrap:** lấy mẫu lại theo cụm Palm ID, các ảnh cùng ID đi cùng nhau.
- **Mean ± std ba seed:** độ lệch chuẩn mẫu ddof=1, không phải CI probe.
- **Delta accuracy:** dùng điểm phần trăm, phân biệt với phần trăm tương đối.

Raw metric thường lưu tỷ lệ 0–1; bảng chuyển sang %. Không lấy trung bình hai
cận CI từng seed để tạo CI cho mean ba seed, hoặc gộp thành 18.000 probe độc lập.
Chưa có ánh xạ Person ID để bootstrap theo người thay vì Palm ID.

## 6. Thời gian và giới hạn kết luận

Phân biệt ROI, trích feature, matching, model loading và I/O. Thời gian cả batch
chia số ảnh không tương đương median/P95 từng ảnh; không cộng median các khâu
thành median tổng. Mỗi bảng timing cần nêu phần cứng, luồng, batch, warm-up và
phạm vi công đoạn.

Kết quả phản ánh Tongji và giao thức đã mô tả, chưa chứng minh tổng quát cho
camera, nền hoặc tư thế tự do. Pilot 10 lớp không so trực tiếp với benchmark
600 lớp. CI chứa 0 không chứng minh hai phương pháp tương đương; accuracy cao
hơn trên tập nhỏ không tự nó chứng minh khác biệt có ý nghĩa thống kê.
