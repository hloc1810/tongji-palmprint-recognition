# Tongji Palmprint Recognition

Đồ án Nhận dạng mẫu: nhận dạng lòng bàn tay không tiếp xúc **closed-set 1:N, 600 Palm ID** trên Tongji Contactless Palmprint Dataset. Gallery/huấn luyện từ session1, đánh giá trên session2. Mỗi bàn tay là một lớp, không đồng nhất Palm ID với Person ID.

## Kết quả hiện tại

Thực nghiệm trên **6.000 ảnh session1 và 6.000 probe session2**, cùng Official ROI 128×128 và manifest. Các số dưới đây là kết quả đã chạy, không phải số liệu từ bài báo hoặc smoke test.

| Pipeline | Rank-1 | Rank-5 | Macro-F1 |
|---|---:|---:|---:|
| M1 — CompCode + 1-NN | 94.43% | 96.97% | 93.56% |
| M2 — CompCode + Linear SVM OvR | 97.37% | 98.52% | 96.82% |
| M3 — CompCode + CRC_RLS | 98.27% | 99.08% | 97.99% |
| M4-E1 — ResNet18 logits | 64.36% ± 1.04 | 81.91% ± 0.65 | 60.79% ± 1.10 |
| M4-E2 — ResNet18 Nearest Cosine | 73.74% ± 0.62 | 87.50% ± 0.44 | 70.51% ± 0.81 |
| M4-E2 — ResNet18 Centroid Cosine | 68.83% ± 0.76 | 83.70% ± 0.63 | 65.37% ± 0.82 |

- M1–M3 dùng chung **CompCode histogram 600-D**, L2 normalization. Chọn tham số bằng 5-fold session1: 1-NN k=1, LinearSVC C=1, CRC λ=0,1. CRC dùng residual Euclidean **không chia chuẩn hệ số lớp**.
- CNN dùng ImageNet-pretrained ResNet18, chọn epoch trên fold0 session1 rồi khởi tạo lại và refit toàn session1. Kết quả CNN là **mean ± sample std** qua seed **17, 29, 43**, ddof=1; std theo điểm phần trăm. Không chọn seed tốt nhất và không ghép ensemble.
- Embedding dùng lại checkpoint của từng seed: **ResNet18 → 512-D → L2 Normalize → cosine**. Nearest lấy similarity lớn nhất trong 10 template/lớp; Centroid lấy trung bình embedding theo lớp rồi chuẩn hóa lại. Không train thêm metric loss.
- Hai biến thể cosine là **đánh giá hậu nghiệm/exploratory**, được bổ sung sau khi đã xem kết quả session2. Mức tăng mean Rank-1 so với logits: Nearest **9,38** và Centroid **4,47 điểm phần trăm**, là chênh lệch mô tả.

Chi tiết: [benchmark đầy đủ](reports/TONGJI_BENCHMARK.md), [CNN ba seed và so sánh thống kê](reports/CNN_EXPERIMENTS.md), [bảng từng run](reports/tables/tongji_benchmark.csv), [mean/std theo pipeline](reports/tables/tongji_benchmark_summary.csv), [lưới CV](reports/tables/tongji_cv_grid_summary.csv).

CPU Ryzen 7 5800H, một luồng: median từng ảnh M1 **11,00 ms**, M2 **3,71 ms**, M3 **14,71 ms**, CNN logits **48,54–48,90 ms** theo seed. Bao gồm đọc ROI, tiền xử lý/trích đặc trưng và tính score; model đã nạp sẵn, chưa gồm Top-k/UI. Embedding mới có thời gian batch chia bình quân, **chưa đo median/P95 từng ảnh**, nên không so trực tiếp với các số này.

## Cài đặt

Khuyến nghị Python **3.11 hoặc 3.12**. Chạy lệnh tại thư mục gốc repo.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,demo]"
# Cài thêm CNN khi cần, chọn PyTorch phù hợp CPU/GPU:
python -m pip install -e ".[cnn]"
```

Linux/macOS: dùng `python3 -m venv .venv` và `source .venv/bin/activate`, các lệnh còn lại tương tự. CUDA được chọn khi có sẵn; `--device cpu` chạy CPU. DirectML trên Windows cần môi trường riêng có `torch-directml`; xem [hồ sơ thực nghiệm CNN](reports/CNN_EXPERIMENTS.md).

## Chuẩn bị dữ liệu

Repo private này lưu mã nguồn, dữ liệu ROI, gallery, trọng số và kết quả thực nghiệm. Không đưa cache, môi trường venv hoặc runtime cục bộ lên repo. Quyền sử dụng Tongji vẫn theo điều kiện của [tác giả dataset](https://cslinzhang.github.io/ContactlessPalm/). Không đổi repo thành public khi chưa rà soát quyền phân phối.

```bash
git clone https://github.com/hloc1810/tongji-palmprint-recognition.git
```

```text
data/raw/
├── session1/      # 00001.bmp … 06000.bmp
└── session2/      # 00001.bmp … 06000.bmp
```

Trong mỗi session, 00001–00010 thuộc Palm ID 1, …, 05991–06000 thuộc Palm ID 600. Đầu vào phải là ROI grayscale 128×128; **không resize ảnh toàn bàn tay thành ROI**.

```bash
palmprint prepare --data data/raw
palmprint eda
palmprint features
```

`prepare` tạo manifest, hash và 5 fold trên session1. Strict audit yêu cầu 600 ID × 10 ảnh/session, kiểm tra ảnh trùng và toàn vẹn dữ liệu.

## Chạy mô hình cổ điển

```bash
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
```

CV/final train chỉ dùng session1; phải khóa cả ba model trước khi đọc feature session2. Artifact và kết quả test không được ghi đè. Nghiên cứu mới cần workspace mới và protocol riêng; không xóa kết quả cũ để tune lại test.

## ResNet18 và embedding matching

Huấn luyện theo [configs/resnet18.yaml](configs/resnet18.yaml). Chốt cấu hình và cả ba seed trước khi đánh giá CNN.

```powershell
foreach ($seed in 17, 29, 43) {
    palmprint-cnn train --seed $seed --device cpu
}
foreach ($seed in 17, 29, 43) {
    palmprint-cnn evaluate --seed $seed --device cpu
}
```

Pretrained weights có thể tải từ PyTorch ở lần train đầu tiên. `--weights none` chỉ phục vụ kiểm thử/offline, không đại diện kết quả transfer learning đã báo cáo. Lệnh `palmprint lock` chỉ khóa classical; hồ sơ khóa CNN của thực nghiệm này được lưu riêng tại workspace cục bộ.

Từ checkpoint đã huấn luyện, tạo gallery session1 và đánh giá hai chiến lược:

```powershell
foreach ($seed in 17, 29, 43) {
    palmprint-cnn embedding-gallery --seed $seed --device cpu
    palmprint-cnn embedding-evaluate --seed $seed --strategy both --device cpu
}
```

Định nghĩa công thức, cache và các lệnh chẩn đoán: [Embedding matching](docs/EMBEDDING_MATCHING.md). `palmprint report` kiểm chứng từng run trước khi xuất bảng; bảng benchmark tổng hợp đã công bố được lưu riêng trong `reports/tables/`.

## Demo và kiểm thử

```bash
streamlit run app/main.py
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q
python scripts/run_smoke.py
```

Demo chọn một model/lượt, trả Palm ID và Top-5 từ artifact cục bộ. Smoke test dùng dữ liệu tổng hợp, không có giá trị ước lượng accuracy Tongji. GitHub Actions kiểm tra Python 3.11/3.12 và cài PyTorch CPU để chạy kiểm thử CNN.

## Cấu trúc dự án

```text
src/palmprint/
├── cli.py, cli_cnn.py       # Điểm vào CLI
├── data/                   # ROI, manifest, audit và folds
├── features/               # CompCode và cache đặc trưng
├── models/                 # Classical, ResNet18, runtime CNN dùng chung
├── training/               # CV, final fit, khóa và fine-tuning
├── evaluation/             # Metrics, latency, logits và cosine matching
└── reporting/              # Kiểm chứng evidence, bảng và biểu đồ
app/                        # Demo Streamlit
configs/                    # Cấu hình thí nghiệm
scripts/                    # Smoke test và audit ROI
tests/                      # Unit và integration tests
docs/                       # Phương pháp và hướng dẫn
reports/                    # Benchmark và bảng kết quả đã công bố
experiments/                # Thiết kế và môi trường thực nghiệm
notebooks/                  # Đọc manifest và phân tích kết quả
```

Không dùng `.gitignore` ở gốc. Repo private giữ dữ liệu, metadata, artifact và kết quả thực nghiệm; cache, `.venv311/`, `.venv312/`, `.runtime/` và các thư mục kiểm thử tạm được loại khỏi bản push. Hãy tạo venv mới theo hướng dẫn cài đặt khi dùng repo. Các thư mục này vẫn được giữ nguyên tại máy thực nghiệm. Nội dung liên kết ra ngoài workspace không được sao chép vào repo.

## Tài liệu và giới hạn

- [Tổ chức code](docs/CODE_STRUCTURE.md), [thuật toán](docs/ALGORITHMS.md), [huấn luyện](docs/TRAINING_GUIDE.md).
- [Protocol](docs/EXPERIMENT_PROTOCOL.md), [tái lập](docs/REPRODUCIBILITY.md), [kiểm thử](docs/QUALITY_ASSURANCE.md), [trạng thái dự án](docs/PROJECT_STATUS.md).
- [Dữ liệu và quyền riêng tư](docs/DATA_PRIVACY.md), [kế hoạch nghiên cứu](docs/RESEARCH_PLAN.md), [hồ sơ Research correctness ban đầu](docs/RESEARCH_CORRECTNESS.md).
- [Đề cương báo cáo](reports/report/REPORT_TEMPLATE.md) và [đề cương slide](reports/slides/SLIDE_OUTLINE.md) vẫn cần hoàn thiện thành sản phẩm học thuật cuối kỳ.

CompCode là bản Python **paper-inspired**, chưa xác minh tương đương số học với MATLAB tác giả. Các kết quả chỉ áp dụng cho dataset/protocol/cấu hình đã ghi nhận, chưa chứng minh nhận dạng người lạ, ảnh điện thoại hoặc webcam. Bootstrap gom theo Palm ID; thiếu ánh xạ Person ID để gom hai tay cùng người.

Mã nguồn hiện tại được dọn sau benchmark. Hash của source và dependency dùng khi chạy được ghi trong hồ sơ thực nghiệm, không được thay bằng hash mới để hợp thức hóa kết quả cũ. Các source snapshot, checkpoint và raw predictions được giữ nguyên trong bản lưu workspace. Xem [tái lập và thay đổi mã nguồn](docs/REPRODUCIBILITY.md).

## Tham khảo

- L. Zhang et al., *Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach*, Pattern Recognition 69 (2017), 199–212. [Dataset và mã tham chiếu](https://cslinzhang.github.io/ContactlessPalm/).
- R. O. Duda, P. E. Hart, D. G. Stork, *Pattern Classification*, 2nd ed., Wiley.
