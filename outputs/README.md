# Bằng chứng thực nghiệm

| Thư mục | Nội dung được giữ |
|---|---|
| `cv/` | CV, out-of-fold predictions và lựa chọn M1–M3 |
| `runs/` | Metric, prediction, score, confusion và CMC từng run M1–M4 |
| `tongji_benchmark/` | Giao thức, môi trường và source snapshot classical |
| `cnn_experiments/` | Giao thức, chẩn đoán và source snapshot CNN |
| `embedding_benchmark/` | Biên bản kiểm chứng embedding và script kiểm tra lịch sử |
| `feature_ablation/` | E3: bảng, bootstrap và hình dẫn xuất |
| `robustness_e6/` | Phân tích độ nhạy của model khóa |
| `spatial_ablation_e3a/` | Global/spatial histogram trên Session 1 |
| `palmnet_m5/` | Giao thức và kết quả M5, cache lớn chỉ lưu cục bộ |
| `roi_comparison/` | 200 mẫu ghép cặp, landmark, ROI, prediction và minh họa |
| `research_correctness/` | Hồ sơ kiểm tra phần mềm/dataset lịch sử |

Các file trong thư mục này là **bằng chứng theo lần chạy**, không phải tất cả
đều là code hiện hành. Đặc biệt, `source_snapshot/` và runner CNN lịch sử được
giữ để truy vết, không nhập ngược vào package. Entry point đang duy trì nằm
trong [scripts](../scripts/README.md).

Không có toàn bộ cache/score/model lớn trên GitHub. Xem phạm vi hiện vật của
từng báo cáo trước khi xác minh hash. Test mới và lần chạy mới ghi vào `.local/`
hoặc thư mục riêng, không tạo thêm `*_tmp` cạnh bằng chứng final.
