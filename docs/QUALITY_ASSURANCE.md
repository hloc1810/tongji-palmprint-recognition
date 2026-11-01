# Kiểm thử và phạm vi xác minh

## Kiểm tra trước khi đưa mã lên GitHub

```bash
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q
python scripts/run_smoke.py
```

CI chạy Python 3.11/3.12, cài PyTorch/torchvision CPU và lưu JUnit/dependency versions. DirectML là backend tùy chọn trong môi trường riêng; test tương ứng được skip khi không có backend.

Đợt dọn mã nguồn: Python **3.11 và 3.12 đều đạt 57 passed, 1 skipped** (thiếu DirectML trong hai môi trường CPU). Smoke test tổng hợp, Ruff lint và format đều đạt. Kiểm thử dùng OMP/MKL/OpenBLAS một luồng; thư mục tạm riêng để tránh xung đột quyền tài khoản Windows. Các lượt thử lỗi môi trường không được ghi là test đã đạt.

Môi trường DirectML riêng đạt **9 passed** cho tests/test_training.py và tests/test_embedding.py, gồm vòng train/refit/evaluate trên DirectML và CPU. Có warning AdamW fallback CPU đã được ghi nhận từ benchmark CNN; không có test thất bại.

## Phạm vi tự động

- Dataset: nhãn Palm ID, kích thước Official ROI, hash toàn vẹn, fold cân bằng và chống đọc session2 trong CV/train.
- CompCode và classical: Gabor/histogram, nghiệm ridge đối chiếu trực tiếp, 1-NN/SVM, khóa chung, chống sửa cấu hình/model hoặc ghi đè test.
- CNN: kiến trúc, hai giai đoạn train/refit trên fixture, AMP fallback, accumulation, checkpoint portable, CLI YAML, Windows dataset pickling và backend.
- Embedding: normalize, nearest/centroid class scoring, gallery cache, frozen checkpoint, evaluate/predict và tích hợp report trên fixture.
- Reporting: tái tính từ all-class scores, nhãn/rank/Top-5, F1, CMC, confusion, bootstrap CI và phát hiện sửa prediction.
- Repository: đường dẫn tài liệu và cấu trúc package.

Smoke test chỉ dùng ảnh tổng hợp; accuracy fixture không phải accuracy Tongji. Test kỹ thuật không thay thế thực nghiệm pretrained hoặc đánh giá trên dữ liệu thật.

## Bằng chứng thực nghiệm riêng

Strict audit 12.000 ROI, benchmark classical, CNN pretrained ba seed và cosine matching đã được chạy. Số liệu và giới hạn nằm trong [benchmark](../reports/TONGJI_BENCHMARK.md) và [báo cáo CNN](../reports/CNN_EXPERIMENTS.md).

Hồ sơ cục bộ: outputs/research_correctness/, outputs/tongji_benchmark/, outputs/cnn_experiments/, outputs/embedding_benchmark/ và outputs/repository_cleanup/. Các thư mục này không được commit vì có metadata máy, source snapshot và dữ liệu dẫn xuất cần giữ riêng.

CNN gốc có khác biệt làm tròn ở trường score gap float32 so với verifier float64. Báo cáo CNN mô tả verification_view đã dùng để kiểm chứng; không sửa raw evidence trong đợt dọn code. CLI report với raw logits cũ vẫn có thể dừng ở trường này.

## Giới hạn

Chưa xác minh parity MATLAB, chưa kiểm thử UI tự động, chưa chứng minh hiệu năng ngoài Tongji, và chưa đo latency từng ảnh cho embedding. Các file đề cương báo cáo/slide vẫn cần hoàn thiện.
