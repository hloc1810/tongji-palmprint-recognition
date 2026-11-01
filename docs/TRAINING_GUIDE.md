# Quy trình huấn luyện và kiểm định mô hình

## 1. Phạm vi, dữ liệu và nhãn

- Bài toán: **closed-set identification**, 600 Palm ID, mỗi Palm ID là một lòng bàn tay (không đồng nhất với người).
- Đầu vào: **Tongji Official ROI**. Ảnh gốc chỉ dùng nếu có nghiên cứu ROI riêng, không trộn với main protocol.
- Nếu bộ dữ liệu đầy đủ: 6.000 ảnh/session1 (gallery/train), 6.000 ảnh/session2 (probe/test); mỗi lớp 10 ảnh mỗi session.
- `palmprint prepare --data data/raw` tạo `manifest.csv`, `fold_assignments.csv` và `audit.json`. Strict mode kiểm tra đủ mẫu/lớp, UID và hash cross-session.
- Mỗi lớp trong session1 được chia **5 fold** (mỗi fold 2 ảnh/lớp) bằng seed=42. Mỗi lượt CV: 4.800 train, 1.200 validation.
- **Session2 không được dùng** trong việc chọn feature, hyperparameter, epoch, mô hình hoặc seed. Pilot HOG trước đây từng quan sát một phần session2: phải ghi rõ giới hạn này trong báo cáo.
- **Không gộp Original Images với Official ROI**: chúng mô tả cùng ảnh chụp dưới hai biểu diễn.
- Giảng viên phải duyệt nguồn dữ liệu theo đề bài. Mã nguồn không chứa bộ ROI thương mại/sinh trắc học.

## 2. Mô hình cổ điển: cùng CompCode Histogram

### 2.1. Extractor

`CompCodeExtractor` mã hóa hướng lọc Gabor, tính histogram theo block và đưa thành vector cố định, chuẩn hóa theo `configs/compcode.yaml`. Extractor không có tham số học từ dữ liệu; cache có hash của manifest và cấu hình. Các tham số Gabor/block được xác định trước qua paper và kiểm chứng bằng validation. **Bản cài đặt Python chưa được chứng minh đồng nhất số học với MATLAB chính thức**; không ghi “reproduction paper” khi chưa so tensor trung gian.

### 2.2. Ba classifier

| Thuật toán | Fit/train thực tế | Chọn trên 5-fold session1 | Artifact |
|---|---|---|---|
| **1-NN** | Dựng gallery feature theo class; không có gradient descent | `k=1`, khoảng cách Euclidean bình phương | `knn.joblib` |
| **LinearSVM (OvR)** | `LinearSVC(loss='squared_hinge', dual=False)` | `C in {0.1,1,10}`, chọn mean Rank-1 CV | `svm.joblib` |
| **CRC_RLS** | Giải hệ ridge bằng Cholesky trong không gian feature | `lambda in {0.1,1.35,10}`, chọn mean Rank-1 CV | `crc.joblib` |

Không gọi 1-NN/CRC là “train qua epoch”; đây là thuật toán xây dựng gallery hoặc fit hệ tuyến tính. **Không thêm SMOTE, class weighting** vì tập chuẩn cân bằng 10 ảnh/lớp/session; chỉ dùng nếu kiểm toán dữ liệu cho thấy thiếu mẫu và định nghĩa lại giao thức.

### 2.3. Lệnh thực thi

```bash
palmprint prepare --data data/raw
palmprint eda
palmprint features
palmprint cv --model knn
palmprint cv --model svm
palmprint cv --model crc

# Validation-only feature ablation, không dùng session2 để chọn feature:
palmprint cv --model svm --representation hog
palmprint cv --model crc --representation global

# Khóa tham số, fit lại với toàn session1:
palmprint train --model knn
palmprint train --model svm
palmprint train --model crc
palmprint lock

# Chỉ sau khi khóa đầy đủ các quyết định:
palmprint evaluate --model knn
palmprint evaluate --model svm
palmprint evaluate --model crc
```

CV xuất `outputs/cv/<rep>_<model>.csv`, `<rep>_<model>_selection.json` và `<rep>_<model>_oof_predictions.csv` của cấu hình thắng trên session1. `train` lưu `.joblib` và `*_freeze.json` có hash, kiểm tra không thay đổi code/config/CV/manifest/ảnh gốc đã audit. Lệnh `lock` khóa đồng thời k-NN, SVM và CRC_RLS trước khi mở ảnh session2. `evaluate` chặn chạy trước `lock` và chặn ghi đè khi đã có `metrics.json`. Kết quả cuối gồm all-class score matrix, thứ hạng thật của mỗi probe, bảng per-class, full CMC và confusion matrix. `report` đối chiếu lại các bảng với score matrix trước khi vẽ biểu đồ. Cả ba classifier luôn được đánh giá trên cùng class 1..600; top-5 là **5 Palm ID phân biệt**.

**Phân biệt ablation và mô hình cuối:** hiện lệnh `train` luôn lấy cấu hình CompCode chuẩn; `--representation hog/global` ở lệnh `cv` chỉ là thử nghiệm trên validation, không tự động đổi mô hình production. Muốn chọn feature khác làm phương pháp chính, phải sửa/gắn rõ pipeline và đóng băng cấu hình **trước khi xem test**.

## 3. CNN ResNet18: quy trình hai giai đoạn

### 3.1. Dữ liệu và augment

- Backbone ResNet18 từ `torchvision`, khởi tạo `ImageNet1K_V1` để transfer learning. `--weights none` chỉ cho smoke/offline hoặc thí nghiệm train-from-scratch có ghi riêng.
- ROI grayscale -> lặp thành 3 kênh -> resize 224x224 -> ImageNet mean/std.
- **Train augment chỉ trong session1 train**: rotation ±5°, brightness/contrast ±8%. Không flip tay trái/phải, không biến đổi quá mạnh làm mất cấu trúc.
- **Validation** không augment, không shuffle; **session2 không tham gia training hoặc chọn epoch**.
- Chọn số epoch và stage dựa trên **fold0 của session1**, 4.800 train + 1.200 val. Không nói đây là 5-fold cho CNN.

### 3.2. Stage A — linear probe / head-only

1. Khóa (`requires_grad=False`) toàn bộ backbone; thay `fc` thành 600 đầu ra, chỉ cập nhật lớp `fc`.
2. Loss: CrossEntropy; `label_smoothing` mặc định 0 (tham số cấu hình khi có bằng chứng cần regularization).
3. Optimizer: AdamW, `head_lr=1e-3`, weight_decay=`1e-4`.
4. Scheduler: cosine decay theo ngân sách số epoch (mặc định 8), dùng cùng công thức trong refit.
5. Early stopping: tốt nhất theo validation Rank-1; nếu hòa, ưu tiên validation CE loss thấp hơn. Patience mặc định 3 epoch.
6. Lưu trọng số **best head epoch** và huấn luyện tiếp stage B từ checkpoint head tốt nhất, không từ epoch cuối.

### 3.3. Stage B — partial fine-tuning

1. Mở `layer4` và `fc`, các layer còn lại không cập nhật.
2. `layer4_lr=1e-4`, `fc_lr=3e-4`: phân nhóm optimizer để lớp classifier thích ứng nhanh hơn backbone.
3. Giữ **BatchNorm running statistics cố định** trong cả hai stage, thích hợp khi mẫu/class hạn chế. Trọng số BN affine ở layer4 vẫn học nếu trainable.
4. Chọn best epoch chung giữa Stage A/B theo Rank-1, phá hòa bằng val loss; không có giả định fine-tune chắc chắn tốt hơn head-only.

### 3.4. Kỹ thuật ổn định và hiệu năng

- **AMP CUDA** (`torch.autocast` + `torch.amp.GradScaler`) tùy chọn; CPU chạy float32. Gradient được `unscale_` trước khi clip.
- **Gradient clipping** norm mặc định 1.0; tắt bằng `--grad-clip 0`.
- **Gradient accumulation** `--accumulate N`, effective batch size là `batch_size*N` trừ cửa sổ cuối nếu không đủ batch. Gradient được chia chính xác theo số microbatch trong cửa sổ.
- **Reproducibility**: seed Python/NumPy/PyTorch/CUDA, DataLoader generator seed, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Vẫn có thể còn bất định theo phần cứng/CUDA và toán tử.
- **Checkpoint**: `outputs/runs/resnet18_seedN/best_dev.pth` được lưu khi validation cải thiện; `artifacts/resnet18_seedN.pth` là model final. Không có tính năng **resume tại batch/epoch đang dở**: nếu bị ngắt sẽ cần chạy lại seed đó.
- **Metric logging**: CSV được cập nhật sau mỗi epoch, đồng thời log tiến độ trên console. `dev_history.csv` (train/val CE, Rank-1, LR và thời gian từng epoch), `final_history.csv` (train metrics refit), `training_config.json` và freeze metadata chứa hash artifact/manifest.

### 3.5. Refit và đánh giá

Sau khi xác định `head_best_epoch` và `selected_layer4_epoch`, chương trình **khởi tạo ResNet18 lại từ trọng số ban đầu** và huấn luyện trên toàn bộ session1 trong số epoch đã khóa, với **cùng chính sách optimizer/augmentation/scheduler**. Nếu validation không chọn Stage B, bỏ fine-tuning. Tập session2 không dùng trong bước này.

```bash
# PowerShell / bash: ở gốc repo, với torch/torchvision phù hợp môi trường
palmprint-cnn train --seed 17
palmprint-cnn train --seed 29
palmprint-cnn train --seed 43

# GPU thiếu VRAM: batch nhỏ + gradient accumulation
palmprint-cnn train --seed 17 --batch-size 8 --accumulate 4

# CPU/offline smoke ONLY; không thay thế transfer-learning experiment:
palmprint-cnn train --seed 99 --weights none --size 64 --head-epochs 1 \
  --finetune-epochs 0 --batch-size 4 --no-amp

# Đánh giá ba model đã khóa; không chọn seed dựa trên kết quả session2
palmprint-cnn evaluate --seed 17
palmprint-cnn evaluate --seed 29
palmprint-cnn evaluate --seed 43
```

**Lưu ý:** lệnh smoke nhiều dòng dùng dấu `\\` cho bash; trên PowerShell viết trên một dòng hoặc dùng backtick. Cấu hình mặc định được đọc trực tiếp từ `configs/resnet18.yaml`; cờ CLI ghi đè trường tương ứng. `--workspace` đặt *trước* subcommand khi cần làm việc trong thư mục khác.

Không dùng validation làm tham số động của final refit (ví dụ ReduceLROnPlateau dựa trên validation trên tập không còn tồn tại). Cosine schedule định trước làm cho lịch huấn luyện final tái lập từ tham số đã chọn.

## 4. Thực nghiệm, thống kê và chống leakage

- **Mô hình cổ điển:** thống kê cross-validation 5 fold; nếu so đặc trưng, giữ classifier và folds cố định. Chỉ báo cáo session2 sau khi mọi quyết định được khóa.
- **CNN:** nếu có tài nguyên, dùng 3 seeds định sẵn (17,29,43), báo cáo từng run và mean/std của ba run; tuyệt đối không lấy seed đạt test cao nhất làm kết quả chính. 3-seed std phản ánh nhạy với ngẫu nhiên, **không** thay cho khoảng tin cậy lấy mẫu.
- **Metrics:** Rank-1, Rank-5, Macro-F1, CMC, confusion matrix, latency CPU cùng điều kiện; lưu thêm P95 khi benchmark có đủ lần đo.
- **Uncertainty:** bootstrap theo Palm ID cho CI Rank-1; đối với so sánh hai mô hình dùng paired bootstrap theo Palm ID; lưu phương pháp, seed, lần lặp. Bootstrap theo Palm ID không khắc phục đầy đủ liên hệ hai bàn tay cùng người vì thiếu ánh xạ người–tay.
- **Ablation:** CompCode block-wise vs global (cùng CRC), CompCode vs HOG (cùng SVM), head-only vs partial fine-tuning (val session1). Không phát biểu nhân quả nếu thay nhiều yếu tố cùng lúc.
- **Robustness:** kiểm tra blur/brightness/rotation tổng hợp theo cấu hình khóa trước; không dùng để chọn mô hình sau khi đã xem test.
- **Data privacy:** không commit file ROI hoặc ảnh sinh trắc học; không lưu ảnh upload người dùng trong artifact/demo.
- **Giới hạn độc lập test:** Một thí nghiệm HOG ban đầu sử dụng một phần session2; thông tin này cần được công bố trong báo cáo. Việc khóa thí nghiệm hiện tại không biến toàn bộ session2 trở thành tập hoàn toàn chưa từng xem.

## 5. Kiểm thử và điều kiện nghiệm thu

```bash
pip install -e '.[test,demo]'
pip install -e '.[cnn]'  # khi cần chạy CNN
pytest -q
python scripts/run_smoke.py
```

Các tiêu chí kiểm thử: gán nhãn và folds đúng, không có ảnh test trong train/val, class ranking nhất quán, CRC được kiểm nghiệm bằng nghiệm ridge tham chiếu, artifact freeze/hash, CNN save-load bằng `weights_only=True`, gradient accumulation, CLI đọc config, dự đoán không tra tên ảnh để lấy đáp án.

**Trạng thái thực nghiệm:** mã và quy trình phải được xác minh bằng smoke tests trên dữ liệu giả lập; không có bằng chứng Rank-1 Tongji thật trước khi được cung cấp dữ liệu và chạy benchmark. Chỉ khi có thực nghiệm trên Tongji thật và bảng kết quả, nhóm mới được viết kết luận so sánh trong báo cáo.
