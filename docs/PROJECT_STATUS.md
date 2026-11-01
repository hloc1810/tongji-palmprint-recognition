# Trạng thái dự án

## Đã hoàn thành

- Strict audit cục bộ: 12.000 Official ROI, 600 Palm ID, 10 ảnh/ID/session.
- M1–M3: CompCode 600-D, chọn tham số bằng 5-fold session1, final fit và test session2 trên artifact đã khóa.
- ResNet18 pretrained: train/refit ba seed 17, 29, 43; thống kê mean/std và đối chiếu ghép cặp theo Palm ID với classical.
- Hai biến thể embedding 512-D: Nearest Cosine và Centroid Cosine, cùng checkpoint, gallery chỉ từ session1, không train thêm.
- Có Rank-1, Rank-5, Macro-F1, CI từng run, CMC/confusion và raw evidence ở workspace cục bộ.
- Có latency CPU từng ảnh cho classical/logits; embedding mới có timing batch chia bình quân.

[Bảng benchmark hiện tại](../reports/TONGJI_BENCHMARK.md) là nguồn số liệu chính. [Báo cáo CNN](../reports/CNN_EXPERIMENTS.md) mô tả ba seed và phạm vi so sánh thống kê.

## Chưa hoàn thành hoặc chưa xác minh

- Chưa kiểm chứng parity số học với MATLAB CR_CompCode tác giả.
- Chưa đo median/P95 từng ảnh cho embedding matching.
- Chưa train metric learning loss, chưa thực hiện ablation mới để xác định nguyên nhân CNN kém cross-session.
- Chưa có Gaussian Naive Bayes, Random Forest hoặc Raw Pixel 1-NN benchmark có bằng chứng.
- Đề cương báo cáo và slide chưa phải bài nộp hoàn chỉnh. Demo chỉ chọn một model/lượt; chưa có kiểm thử UI tự động.
- Cần hoàn tất nguồn trích dẫn và minh chứng quyền sử dụng dữ liệu trong bài nộp.

## Diễn giải nghiên cứu

Cosine matching là so sánh hậu nghiệm sau khi đã quan sát kết quả session2. Không chọn seed/chiến lược hoặc tiếp tục tune theo test. Không có bằng chứng xác nhận pilot HOG/120 Palm ID được nhắc trong tài liệu khởi đầu; không trình bày chuỗi warning kế thừa là lịch sử thực nghiệm đã xác minh.

Mã đã được tổ chức lại sau các lần chạy. Giữ nguyên raw metrics, checkpoint, source snapshot và bảng benchmark; không sửa khóa/hash của thí nghiệm cũ. Xem [tái lập](REPRODUCIBILITY.md), [kiểm thử](QUALITY_ASSURANCE.md) và [cấu trúc code](CODE_STRUCTURE.md).
