# Tổ chức mã nguồn

## Nguyên tắc phân chia

Mã nguồn chia theo nhiệm vụ, không chia một hàm thành một file. Mỗi module có một vai trò rõ ràng và nằm trong một package tương ứng. Các thư mục `data/`, `features/`, `models/`, `training/`, `evaluation/`, `reporting/` là mã Python trong `src/palmprint/`, khác với các thư mục dữ liệu hoặc báo cáo ở gốc repository.

| Package | Vai trò | File chính |
|---|---|---|
| `data` | Đọc ảnh ROI, nhãn, kiểm toán dataset, manifest và folds | `dataset.py`, `preparation.py` |
| `features` | CompCode, biến đổi đặc trưng và cache tách theo session | `compcode.py`, `extraction.py` |
| `models` | Thuật toán k-NN/SVM/CRC_RLS, kiến trúc ResNet18 | `classical.py`, `resnet.py`, `cnn_runtime.py` |
| `training` | CV, chọn tham số, khóa giao thức, huấn luyện CNN | `selection.py`, `classical.py`, `cnn.py` |
| `evaluation` | Metrics, phân tích sai số và dự đoán CNN | `metrics.py`, `analysis.py`, `cnn.py`, `embedding.py` |
| `reporting` | Bằng chứng scores/nhãn và bảng/hình báo cáo | `evidence.py`, `summary.py`, `cnn_comparison.py` |

## Luồng chạy

```text
Official ROI → data → features → models → training
                                            ↓
                                          evaluation
                                            ↓
                                          reporting
                                            ↓
                                   artifacts / demo / báo cáo
```

M1–M3 dùng chung CompCode histogram; M4 học trực tiếp từ ảnh ROI qua ResNet18. `cli.py` và `cli_cnn.py` gọi các module tương ứng, không chứa thuật toán học máy. `app/main.py` dùng artifact đã huấn luyện. `tests/` kiểm tra cùng API mà CLI sử dụng.

## Giao diện lệnh ổn định

```bash
palmprint prepare --data data/raw
palmprint cv --model knn
palmprint cv --model svm
palmprint cv --model crc
palmprint train --model knn
palmprint train --model svm
palmprint train --model crc
palmprint lock
palmprint evaluate --model knn
palmprint evaluate --model svm
palmprint evaluate --model crc
palmprint report
palmprint-cnn train --config configs/resnet18.yaml
```

`palmprint-cnn train --config configs/resnet18.yaml` đọc cấu hình được chỉ định. Nếu không có `--config`, CLI đọc `configs/resnet18.yaml` tại workspace; các cờ như `--seed`, `--batch-size`, `--weights` ghi đè trường tương ứng. Không dùng session2 để chọn tham số.

## Kiểm tra và tái lập

- `python -m pytest -q` kiểm tra đơn vị và vòng đời classical/CNN trên dữ liệu tổng hợp.
- `python scripts/run_smoke.py` thực hiện quy trình cổ điển end-to-end trên mẫu tổng hợp, không phải kết quả Tongji.
- Kiểm chứng cấu hình, code hash và dữ liệu trước khi đánh giá cross-session.
- Nội dung thí nghiệm và cách diễn giải metrics: [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md), [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md).

## Giới hạn

Benchmark Tongji đã được chạy và được ghi trong reports/. Parity CompCode MATLAB vẫn chưa xác minh; trạng thái chi tiết ở [`PROJECT_STATUS.md`](PROJECT_STATUS.md).


CNN training, logits evaluation và embedding cùng dùng models/cnn_runtime.py cho thiết bị, DataLoader và inference context. evaluation/ không import các helper riêng của training/. Các output snapshot và builder dùng một lần nằm dưới outputs/; theo yêu cầu hiện tại, chúng được lưu cùng toàn workspace trong repo private.
