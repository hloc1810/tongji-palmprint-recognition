# Giai đoạn 2 — ResNet18: ba seed và so sánh với mô hình cổ điển

Hoàn tất: 2026-11-01T14:33:21.525045+07:00 (Asia/Saigon).

Các số liệu dưới đây được đọc từ artifact và dự đoán của lần chạy thật trên Tongji. Không dùng kết quả synthetic hoặc lấy accuracy của tác giả làm kết quả nhóm.

## Thiết kế đã chốt trước đánh giá CNN

- 600 Palm ID, 6.000 ảnh session1 và 6.000 probe session2, cùng manifest với benchmark M1–M3.
- ResNet18 pretrained ImageNet1K V1; ROI grayscale lặp ba kênh, resize trực tiếp 224×224, chuẩn hóa ImageNet; không dùng CompCode cho CNN.
- Seed cố định: [17, 29, 43]; batch 32; tối đa 8 epoch head và 5 epoch layer4; patience=3.
- Chọn stage/epoch trên fold0 session1: 4.800 train + 1.200 val. Ưu tiên val Rank-1, hòa thì CE thấp hơn; sau đó khởi tạo lại và refit toàn bộ 6.000 ảnh session1 theo lịch đã chọn.
- AdamW: head LR=0.001; fine-tune layer4 LR=0.0001, fc LR=0.0003; weight decay=0.0001; cosine LR, clip norm=1.0.
- Giữ BatchNorm running statistics cố định. Train augmentation: rotation ±5°, brightness/contrast ±8%; validation/test không augmentation.
- Đã khóa cả ba model CNN final trước khi đánh giá bất kỳ seed CNN nào trên session2. Không chọn seed tốt nhất bằng test; không ghép ensemble.
- Huấn luyện và đánh giá dùng Radeon RX 5500M qua DirectML, float32. Torch CPU wheel + plugin DirectML vẫn thực thi convolution trên GPU; CUDA/AMP không được dùng.
- Một phép toán AdamW `aten::lerp.Scalar_out` fallback về CPU. Checkpoint lưu tensor CPU; validation DirectML dùng no_grad để tránh lỗi inference tensors của backend.

## Kết quả từng seed trên session2

| Seed | Stage chọn | Epoch head / layer4 khi refit | Best val Rank-1 | Test Rank-1 | Test Rank-5 | Macro-F1 | Đúng / 6.000 | CI 95% Rank-1 |
|---|---|---|---:|---:|---:|---:|---:|---|
| 17 | layer4 | 8 / 5 | 98.9167% | 64.2167% | 81.5500% | 60.4031% | 3853 | [61.1000%; 67.0000%] |
| 29 | layer4 | 8 / 5 | 98.9167% | 63.4000% | 81.5167% | 59.9403% | 3804 | [60.2496%; 66.1338%] |
| 43 | layer4 | 8 / 5 | 98.4167% | 65.4667% | 82.6667% | 62.0364% | 3928 | [62.5500%; 68.1000%] |

CI từng run: bootstrap theo Palm ID, 1.000 lần, seed=42; số probe của mỗi run là 6.000.

## Chẩn đoán khoảng cách lớn giữa session1 và session2

Kết quả test thấp được giữ nguyên. Các kiểm tra dưới đây là chẩn đoán sau thực nghiệm, chỉ inference từ checkpoint đã khóa; không train lại, sửa tham số hoặc chọn seed bằng kết quả test.

| Seed | Rank-1 đánh giá lại 6.000 ảnh session1, không augment | Số ROI đối chiếu CPU/GPU | Top-1 CPU/GPU khớp | Max sai khác logit |
|---|---:|---:|---:|---:|
| 17 | 99.9667% | 1200 | 100.0000% | 0.00019073 |
| 29 | 100.0000% | 1200 | 100.0000% | 0.00019073 |
| 43 | 99.9833% | 1200 | 100.0000% | 0.00018311 |

Tập đối chiếu CPU/GPU gồm ROI đầu tiên của mỗi Palm ID trong mỗi session: 600 ảnh/session, 1.200 ảnh/seed. Checkpoint nhận dạng tốt session1 và CPU/GPU nhất quán trên tập này; chưa thấy bằng chứng lỗi chuyển checkpoint hoặc backend gây ra mức giảm Rank-1 khoảng 35 điểm phần trăm. Kết quả cho thấy khả năng tổng quát hóa cross-session kém của cấu hình hiện tại. Chưa xác định nguyên nhân duy nhất nếu chưa có các ablation độc lập; không dùng kết quả này để tự đổi cấu hình và đánh giá lại test.

## Kiểm chứng số học và nguồn gốc bảng dẫn xuất

Verifier nghiêm ngặt dừng ở trường score_gap_top2_minus_top1: writer tính phép trừ float32, trong khi score matrix được lưu dưới dạng cast float64 và verifier tính phép trừ float64. Đây là trường chênh lệch score dẫn xuất, không phải nhãn dự đoán hoặc chỉ số accuracy.
Sai khác lớn nhất ở sáu run là 9.536743164e-07; tất cả đều nằm trong giới hạn làm tròn float32 đã kiểm tra theo từng hàng.
Không sửa file gốc. Bản verification_view riêng tính lại duy nhất gap bằng float64 từ score matrix gốc, tái sinh errors.csv dẫn xuất từ cùng prediction rồi chạy toàn bộ verifier có sẵn. Nhãn, thứ hạng, score matrix, metric, CI và lựa chọn model không thay đổi. Bảng evidence_verification.csv lưu cả hash prediction gốc và hash prediction của view để truy vết. CLI report nguyên bản trên raw float32 vẫn có thể dừng ở kiểm tra gap này; quy trình hậu xử lý được lưu riêng để bảo toàn source đã khóa.
Bằng chứng chi tiết: outputs/cnn_experiments/float32_gap_verification_audit.json, verification_view/ và checkpoint_diagnostics.json.

## Tổng hợp ba seed

| Chỉ số | Mean ± sample std (điểm phần trăm) | Số seed |
|---|---:|---:|
| rank1 | 64.3611% ± 1.0409 | 3 |
| rank5 | 81.9111% ± 0.6545 | 3 |
| macro_f1 | 60.7933% ± 1.1012 | 3 |

Std dùng ddof=1, mô tả biến thiên giữa ba lần train. Nó không phải CI của quần thể test. Mean là trung bình chỉ số của ba model riêng biệt; không phải accuracy của một ensemble hoặc một model được chọn bằng test.

## So sánh trên cùng 6.000 probe

| Pipeline | Rank-1 | Rank-5 | Macro-F1 |
|---|---:|---:|---:|
| knn + CompCode | 94.4333% | 96.9667% | 93.5583% |
| svm + CompCode | 97.3667% | 98.5167% | 96.8154% |
| crc + CompCode | 98.2667% | 99.0833% | 97.9926% |
| ResNet18 — mean 3 seed | 64.3611% | 81.9111% | 60.7933% |

So sánh chính dùng chênh lệch **mean Rank-1 của ba seed CNN − Rank-1 của baseline**. Mỗi lần bootstrap resample Palm ID một lần, áp dụng cùng probe cho tất cả model/seed; 5.000 lần, seed=42. Không gộp ba run thành 18.000 probe độc lập.

| Baseline | Δ Rank-1 (điểm phần trăm) | CI 95% (điểm phần trăm) | Khoảng Bonferroni cho 3 so sánh, family-wise 95% |
|---|---:|---|---|
| knn | -30.0722 | [-32.8668; -27.3832] | [-33.5519; -26.8424] |
| svm | -33.0056 | [-35.8113; -30.3554] | [-36.4927; -29.7500] |
| crc | -33.9056 | [-36.6778; -31.2167] | [-37.3260; -30.6851] |

Khoảng bootstrap trên có điều kiện trên ba seed đã train; không bao hàm toàn bộ biến thiên của những seed tương lai. Các khoảng Bonferroni là khoảng biên 98,3333% cho từng so sánh để kiểm soát ba so sánh chính ở mức family-wise xấp xỉ 95%. Bảng so sánh từng seed với baseline được lưu riêng, mang tính mô tả và chưa điều chỉnh đa so sánh.

## Độ trễ CPU cùng điều kiện từng ảnh

| Model | Probe | Tổng median (ms) | Tổng P95 (ms) |
|---|---:|---:|---:|
| knn + CompCode | 6000 | 10.9987 | 20.2279 |
| svm + CompCode | 6000 | 3.7102 | 5.5744 |
| crc + CompCode | 6000 | 14.7064 | 20.8129 |
| resnet18_seed17 | 6000 | 48.5417 | 59.4951 |
| resnet18_seed29 | 6000 | 48.8995 | 60.4324 |
| resnet18_seed43 | 6000 | 48.7241 | 62.6022 |

CPU Ryzen 7 5800H, một luồng, model nạp sẵn, ba lượt warmup, một lượt đo toàn bộ probe cho mỗi model. CNN gồm đọc/giải mã ROI + transform + forward logits; classical gồm đọc/giải mã ROI + CompCode + score lớp. Không gồm model loading, sắp hạng Top-k hoặc UI. Bộ đệm file hệ điều hành có thể đã ấm. Accuracy CNN được đánh giá trên GPU; bảng này chỉ đo thời gian inference CPU của cùng checkpoint. CNN latency từng probe cũng được lưu CSV; thời gian đánh giá batch GPU được ghi riêng, không so trực tiếp với các cột CPU này.

## Diễn giải và giới hạn

- Đây là so sánh toàn pipeline: CNN học feature từ ROI, các model cổ điển dùng CompCode. Không quy toàn bộ chênh lệch cho riêng classifier.
- Classical dùng 5-fold để chọn tham số; CNN dùng fold0 để chọn epoch. Cần ghi rõ khác biệt về chọn mô hình khi báo cáo.
- Kết quả classical session2 đã được quan sát ở giai đoạn 1. Cấu hình CNN được lấy từ cấu hình sẵn có và chốt trước đánh giá CNN, nhưng toàn bộ nghiên cứu là so sánh tuần tự/exploratory, không phải một lần test hoàn toàn chưa từng được xem cho mọi candidate.
- Bootstrap gom theo Palm ID. Thiếu ánh xạ Person ID để gom hai tay cùng chủ thể; chỉ có ba seed nên chưa ước lượng đầy đủ độ nhạy với mọi khởi tạo/hardware/backend.
- Mã nguồn chứa warning kế thừa về pilot HOG, không có bằng chứng tương ứng; người thực hiện xác nhận chưa chạy model trước benchmark giai đoạn 1. Không dùng chuỗi warning đó làm lịch sử thực nghiệm đã được kiểm chứng.
- CNN không dùng CompCode nên không phải một biến thể CR_CompCode. Kết quả của nhóm không được coi là tái hiện accuracy MATLAB tác giả.

## Bằng chứng và tái lập

- Pretrained SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Manifest SHA-256: `99bbbe45a4807d48801a28a6aadfec2efee46a3f4c4fe3d3af8e2337241c8ae0`.
- CNN source SHA-256: `4e5398c5a24a0a56966fd40b1e7ff2a88d979190722e02711bcbe686bd4c6be0`.
- Torch 2.4.1+cpu, torchvision 0.19.1+cpu, DirectML 0.2.5.dev240914; dependency đầy đủ được lưu theo môi trường riêng.
- Cấu hình, source snapshot, hash weights, log, tiến độ, test kỹ thuật: `outputs/cnn_experiments/`.
- Khóa ba seed: `artifacts/cnn_protocol_lock.json`; trọng số/freeze: `artifacts/resnet18_seed17.*`, `seed29.*`, `seed43.*`.
- Raw metric/score/prediction/confusion/CMC/per-class: `outputs/runs/resnet18_seed17/`, `resnet18_seed29/`, `resnet18_seed43/`.
- Tổng hợp: `reports/tables/cnn_seed_results.csv`, `cnn_seed_summary.csv`, `cnn_mean_vs_classical_paired.csv`, `cnn_each_seed_vs_classical_paired.csv`, `cnn_cpu_latency.csv`.
- Biểu đồ: `reports/figures/cnn_training_curves.png`, `cnn_vs_classical_cmc_rank20.png`; các biểu đồ CMC/confusion đầy đủ được sinh riêng cho từng run.
- Các artifact và metric M1–M3 đã được kiểm tra hash, giữ nguyên so với đầu giai đoạn này.

Lệnh đã chạy từ gốc dự án, trong môi trường DirectML có dependency snapshot:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) '.cache/torch'
.runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/run_experiments.py
# Resume postprocessing after the documented float32 score-gap verifier issue:
.runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/checkpoint_diagnostics.py
.runtime/venv-directml/Scripts/python.exe -u outputs/cnn_experiments/finish_experiments.py
.runtime/venv-directml/Scripts/python.exe outputs/cnn_experiments/write_report.py
```

Runner giữ lại các seed đã freeze và kết quả test đã tồn tại; không ghi đè để chọn lại model. Nếu chạy một nghiên cứu khác, dùng workspace/protocol mới và công khai lịch sử sử dụng session2.

## Nguồn tham khảo

- [Torchvision ResNet18 và ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Transfer learning — hướng dẫn PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [PyTorch với DirectML trên Windows — Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows).
