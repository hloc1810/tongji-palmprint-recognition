"""Prepare repository documentation from existing benchmark tables; never rerun models."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(name, text):
    (ROOT / name).write_text(text.strip() + "\n", encoding="utf-8")


with (ROOT / "reports/tables/tongji_benchmark_summary.csv").open(encoding="utf-8") as stream:
    summary = list(csv.DictReader(stream))


def display(row, metric):
    result = f"{100 * float(row[metric + '_mean']):.2f}%"
    deviation = row[metric + "_sample_std_ddof1"]
    return result + (f" ± {100 * float(deviation):.2f}" if deviation else "")


labels = [
    "M1 — CompCode + 1-NN",
    "M2 — CompCode + Linear SVM OvR",
    "M3 — CompCode + CRC_RLS",
    "M4-E1 — ResNet18 logits",
    "M4-E2 — ResNet18 Nearest Cosine",
    "M4-E2 — ResNet18 Centroid Cosine",
]
table = "\n".join(
    "| " + " | ".join([label, *(display(row, m) for m in ("rank1", "rank5", "macro_f1"))]) + " |"
    for label, row in zip(labels, summary, strict=True)
)
write("README.md", f"""
# Tongji Palmprint Recognition

Đồ án Nhận dạng mẫu: nhận dạng lòng bàn tay không tiếp xúc **closed-set 1:N, 600 Palm ID** trên Tongji Contactless Palmprint Dataset. Gallery/huấn luyện từ session1, đánh giá trên session2. Mỗi bàn tay là một lớp, không đồng nhất Palm ID với Person ID.

## Kết quả hiện tại

Thực nghiệm trên **6.000 ảnh session1 và 6.000 probe session2**, cùng Official ROI 128×128 và manifest. Các số dưới đây là kết quả đã chạy, không phải số liệu từ bài báo hoặc smoke test.

| Pipeline | Rank-1 | Rank-5 | Macro-F1 |
|---|---:|---:|---:|
{table}

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
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,demo]"
# Cài thêm CNN khi cần, chọn PyTorch phù hợp CPU/GPU:
python -m pip install -e ".[cnn]"
```

Linux/macOS: dùng `python3 -m venv .venv` và `source .venv/bin/activate`, các lệnh còn lại tương tự. CUDA được chọn khi có sẵn; `--device cpu` chạy CPU. DirectML trên Windows cần môi trường riêng có `torch-directml`; xem [hồ sơ thực nghiệm CNN](reports/CNN_EXPERIMENTS.md).

## Chuẩn bị dữ liệu

Tải **Official ROI Images** từ [trang tác giả Tongji](https://cslinzhang.github.io/ContactlessPalm/) theo điều kiện sử dụng. Ảnh ROI, gallery embedding và trọng số huấn luyện không được phân phối trong repo.

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
foreach ($seed in 17, 29, 43) {{
    palmprint-cnn train --seed $seed --device cpu
}}
foreach ($seed in 17, 29, 43) {{
    palmprint-cnn evaluate --seed $seed --device cpu
}}
```

Pretrained weights có thể tải từ PyTorch ở lần train đầu tiên. `--weights none` chỉ phục vụ kiểm thử/offline, không đại diện kết quả transfer learning đã báo cáo. Lệnh `palmprint lock` chỉ khóa classical; hồ sơ khóa CNN của thực nghiệm này được lưu riêng tại workspace cục bộ.

Từ checkpoint đã huấn luyện, tạo gallery session1 và đánh giá hai chiến lược:

```powershell
foreach ($seed in 17, 29, 43) {{
    palmprint-cnn embedding-gallery --seed $seed --device cpu
    palmprint-cnn embedding-evaluate --seed $seed --strategy both --device cpu
}}
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

`data/raw/`, metadata cục bộ, `artifacts/`, `outputs/`, môi trường Python và cache được Git bỏ qua. Chỉ các bảng tổng hợp và hai hình nghiên cứu được chọn mới đưa vào repo. Các đầu ra thực nghiệm đầy đủ được giữ tại máy để truy vết, không bị xóa khi dọn mã nguồn.

## Tài liệu và giới hạn

- [Tổ chức code](docs/CODE_STRUCTURE.md), [thuật toán](docs/ALGORITHMS.md), [huấn luyện](docs/TRAINING_GUIDE.md).
- [Protocol](docs/EXPERIMENT_PROTOCOL.md), [tái lập](docs/REPRODUCIBILITY.md), [kiểm thử](docs/QUALITY_ASSURANCE.md), [trạng thái dự án](docs/PROJECT_STATUS.md).
- [Dữ liệu và quyền riêng tư](docs/DATA_PRIVACY.md), [kế hoạch nghiên cứu](docs/RESEARCH_PLAN.md), [hồ sơ Research correctness ban đầu](docs/RESEARCH_CORRECTNESS.md).
- [Đề cương báo cáo](reports/report/REPORT_TEMPLATE.md) và [đề cương slide](reports/slides/SLIDE_OUTLINE.md) vẫn cần hoàn thiện thành sản phẩm học thuật cuối kỳ.

CompCode là bản Python **paper-inspired**, chưa xác minh tương đương số học với MATLAB tác giả. Các kết quả chỉ áp dụng cho dataset/protocol/cấu hình đã ghi nhận, chưa chứng minh nhận dạng người lạ, ảnh điện thoại hoặc webcam. Bootstrap gom theo Palm ID; thiếu ánh xạ Person ID để gom hai tay cùng người.

Mã nguồn hiện tại được dọn sau benchmark. Hash của source và dependency dùng khi chạy được ghi trong hồ sơ thực nghiệm, không được thay bằng hash mới để hợp thức hóa kết quả cũ. Các source snapshot, checkpoint và raw predictions cục bộ được giữ nguyên. Xem [tái lập và thay đổi mã nguồn](docs/REPRODUCIBILITY.md).

## Tham khảo

- L. Zhang et al., *Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach*, Pattern Recognition 69 (2017), 199–212. [Dataset và mã tham chiếu](https://cslinzhang.github.io/ContactlessPalm/).
- R. O. Duda, P. E. Hart, D. G. Stork, *Pattern Classification*, 2nd ed., Wiley.
""")

write("docs/PROJECT_STATUS.md", """
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
""")

write("docs/QUALITY_ASSURANCE.md", """
# Kiểm thử và phạm vi xác minh

## Kiểm tra trước khi đưa mã lên GitHub

```bash
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q
python scripts/run_smoke.py
```

CI chạy Python 3.11/3.12, cài PyTorch/torchvision CPU và lưu JUnit/dependency versions. DirectML là backend tùy chọn trong môi trường riêng; test tương ứng được skip khi không có backend.

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
""")

# Consolidate the Vietnamese embedding guide into one canonical document.
embedding = (ROOT / "docs/EMBEDDING_MATCHING_VI.md").read_text(encoding="utf-8")
embedding = embedding[:embedding.index("## Giới hạn đánh giá")]
embedding += """## Kết quả hiện tại và giới hạn

Đã chạy cả ba seed 17, 29, 43 trên Tongji thật. Nearest Cosine đạt mean Rank-1 73,7444% ± 0,6241 điểm phần trăm; Centroid đạt 68,8278% ± 0,7558. Số liệu đầy đủ, CI từng seed và timing nằm trong [bảng benchmark](../reports/TONGJI_BENCHMARK.md).

- Đây là đánh giá hậu nghiệm sau benchmark logits/classical, không phải test set chưa từng quan sát. Không chọn seed hoặc chiến lược theo session2.
- Dùng cùng checkpoint cross-entropy đã có; không train contrastive/triplet/ArcFace loss.
- Centroid là trung bình các embedding đã L2-normalize rồi chuẩn hóa lại. Cả hai chiến lược trả 600 score lớp; Top-5 đếm Palm ID khác nhau.
- Gallery cache gắn checkpoint hash, manifest hash, kích thước ảnh và thứ tự UID/Palm ID. Không phân phối ảnh, gallery hoặc checkpoint trong repo.
- File metrics gốc có warning kế thừa về historical pilots; chưa có bằng chứng xác nhận pilot HOG/120 ID. Điều đã biết là session2 đã được đánh giá trước khi bổ sung embedding.
- Thời gian batch chia số ảnh khác median/P95 từng ảnh. Chưa đo latency từng ảnh cho embedding.
- Chạy lại trong workspace mới nếu cần nghiên cứu khác; evaluator từ chối ghi đè kết quả hiện có.
"""
write("docs/EMBEDDING_MATCHING.md", embedding)
(ROOT / "docs/EMBEDDING_MATCHING_VI.md").unlink()

# Update only stale historical claims in otherwise useful reference documents.
for filename in ["MODEL_VALIDATION.md", "EXPERIMENT_PROTOCOL.md", "RESEARCH_PLAN.md"]:
    document = ROOT / "docs" / filename
    lines = document.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "pilot" not in line.lower():
            continue
        if "120" in line or "notebook HOG+SVM" in line or line.startswith("- **Giới hạn từ pilot:"):
            lines[i] = "**Lịch sử test:** Chưa có bằng chứng xác nhận pilot HOG được nhắc trong bản tài liệu ban đầu. Benchmark classical và CNN đã quan sát session2 trước khi bổ sung embedding; các so sánh bổ sung là hậu nghiệm. Không dùng test để chọn cấu hình hoặc seed."
        elif "- [ ] Nêu giới hạn tính độc lập" in line:
            lines[i] = "- [ ] Nêu lịch sử quan sát session2 và phạm vi hậu nghiệm của các so sánh bổ sung."
        elif line.startswith("| Giới hạn tính độc lập"):
            lines[i] = "| Giới hạn tính độc lập của session2 | Ghi rõ thứ tự classical, CNN logits và embedding; không tiếp tục tune bằng test |"
        elif "lịch sử pilot có xem" in line:
            lines[i] = line.replace("lịch sử pilot có xem một phần test", "các so sánh bổ sung diễn ra sau khi đã quan sát test")
    write(f"docs/{filename}", "\n".join(lines))

structure = ROOT / "docs/CODE_STRUCTURE.md"
text = structure.read_text(encoding="utf-8")
text = text.replace("`classical.py`, `resnet.py`", "`classical.py`, `resnet.py`, `cnn_runtime.py`")
text = text.replace("`metrics.py`, `analysis.py`, `cnn.py`", "`metrics.py`, `analysis.py`, `cnn.py`, `embedding.py`")
text = text.replace("`evidence.py`, `summary.py`", "`evidence.py`, `summary.py`, `cnn_comparison.py`")
text = text.replace("Việc chia module không thay đổi phương pháp khoa học; kết quả thực nghiệm trên dữ liệu Tongji thật và việc tái hiện CompCode MATLAB vẫn cần kiểm chứng theo", "Benchmark Tongji đã được chạy và được ghi trong reports/. Parity CompCode MATLAB vẫn chưa xác minh; trạng thái chi tiết ở")
text += "\n\nCNN training, logits evaluation và embedding cùng dùng models/cnn_runtime.py cho thiết bị, DataLoader và inference context. evaluation/ không import các helper riêng của training/. Các output snapshot và builder dùng một lần chỉ giữ cục bộ dưới outputs/, được loại khỏi Git.\n"
write("docs/CODE_STRUCTURE.md", text)

history = ROOT / "docs/RESEARCH_CORRECTNESS.md"
text = history.read_text(encoding="utf-8")
first, rest = text.split("\n", 1)
write("docs/RESEARCH_CORRECTNESS.md", first + "\n\n> Hồ sơ lịch sử của đợt audit trước benchmark. Các ghi chú ‘chưa chạy’ bên dưới mô tả thời điểm đó. Trạng thái và số liệu hiện tại nằm trong [PROJECT_STATUS.md](PROJECT_STATUS.md) và [benchmark](../reports/TONGJI_BENCHMARK.md).\n" + rest)

repro = (ROOT / "docs/REPRODUCIBILITY.md").read_text(encoding="utf-8")
repro += """

## Source sau benchmark

Đợt dọn repository chỉ chuẩn hóa style/import, gom CNN runtime và cập nhật mô tả. Thuật toán, config, trọng số, raw metrics và bảng benchmark được giữ nguyên. Hash source hiện tại khác hash đã khóa khi chạy; không sửa freeze/protocol_lock để bỏ qua kiểm tra này.

- Muốn kiểm tra lịch sử chính xác: dùng source_snapshot cùng môi trường của đợt chạy, được giữ cục bộ trong outputs/tongji_benchmark/ và outputs/cnn_experiments/.
- Muốn chạy một nghiên cứu mới bằng source hiện tại: tạo workspace mới, chuẩn bị manifest, CV/train/lock và đánh giá theo giao thức mới. Không ghi đè hoặc gọi lại test của workspace đã có kết quả.
- Repo GitHub chỉ chứa mã, config, tài liệu, bảng tổng hợp và hai hình nghiên cứu đã chọn. Checkpoint, gallery, manifest, predictions và log đầy đủ nằm tại máy.
- Dependency snapshot của các lần chạy nằm trong [experiments/environments](../experiments/environments/README.md). Đây là hồ sơ phiên bản Windows/Python 3.11, không phải lockfile dùng chung mọi hệ điều hành.
"""
write("docs/REPRODUCIBILITY.md", repro)

environment_dir = ROOT / "experiments/environments"
environment_dir.mkdir(exist_ok=True)
for source, name in [("tongji_benchmark", "classical.txt"), ("cnn_experiments", "cnn-directml.txt")]:
    lines = (ROOT / f"outputs/{source}/dependencies.txt").read_text(encoding="utf-8").splitlines()
    # The project editable-install entry contains a machine-specific path, not a dependency pin.
    lines = [line for line in lines if line and not line.startswith(("-e ", "#"))
             and "file:" not in line and "D:\\" not in line and "D:/" not in line]
    write(f"experiments/environments/{name}", "\n".join(lines))
write("experiments/environments/README.md", """
# Môi trường thực nghiệm

- classical.txt: dependency snapshot của benchmark M1–M3, Python 3.11.15 trên Windows CPU.
- cnn-directml.txt: dependency snapshot của ResNet18 ba seed, Python 3.11.15, torch 2.4.1/torchvision 0.19.1 và DirectML 0.2.5.dev240914 trên Radeon RX 5500M.

Đã bỏ đường dẫn editable-install của máy khỏi các snapshot. Không coi đây là lockfile đa nền tảng. Các phiên bản tương thích cài đặt chung được khai báo trong pyproject.toml; backend DirectML cần môi trường riêng. Kết quả embedding do người thực hiện chạy với môi trường Python 3.12 cục bộ; không gán snapshot DirectML này cho lượt embedding.
""")
write("experiments/README.md", """
# Hồ sơ thí nghiệm

registry.yaml ghi thiết kế E0–E6. Trạng thái thực tế và số liệu nằm trong [benchmark](../reports/TONGJI_BENCHMARK.md) và [báo cáo CNN](../reports/CNN_EXPERIMENTS.md), không suy ra từ registry rằng mọi ablation đã chạy.

Đã chạy M1–M3, ResNet18 ba seed và hai biến thể cosine từ cùng checkpoint. Cosine được bổ sung sau benchmark logits, là đánh giá hậu nghiệm; không sửa đăng ký cũ thành một kế hoạch được chốt từ trước. Chưa chạy các ablation/robustness thật chỉ vì đã có code hoặc test synthetic.

[Snapshot môi trường](environments/README.md) được đưa vào repo. Source snapshot, protocol, artifacts và raw evidence đầy đủ giữ cục bộ dưới outputs/ và artifacts/. Git repo được tạo sau các thực nghiệm; commit hiện tại không thay thế hash source đã lưu khi chạy.
""")

# Replace a legacy appendix whose Vietnamese text was irreversibly saved as '?'.
benchmark = (ROOT / "reports/TONGJI_BENCHMARK.md").read_text(encoding="utf-8")
marker = "\n---\n\n# K?t qu?"
if marker in benchmark:
    benchmark = benchmark.split(marker, 1)[0]
    with (ROOT / "reports/tables/tongji_benchmark.csv").open(encoding="utf-8") as stream:
        classical = [r for r in csv.DictReader(stream) if r["model"] in ("knn", "svm", "crc")]
    rows = "\n".join(f"| {r['model']} | {100 * float(r['cv_mean_rank1']):.4f}% | {float(r['latency_median_ms']):.4f} | {float(r['latency_p95_ms']):.4f} | {float(r['training_fit_seconds']):.6f} |" for r in classical)
    benchmark += """

## Phụ lục benchmark cổ điển

5-fold session1, seed=42, 4.800 train/1.200 val mỗi fold. SVM C={0,1; 1; 10}, CRC λ={0,1; 1,35; 10}; chọn mean CV Rank-1 cao nhất, hòa chọn tham số nhỏ nhất. SVM chọn C=1; CRC chọn λ=0,1. Tất cả artifact đã khóa trước đánh giá session2.

| Model | Mean CV Rank-1 | CPU median (ms/ảnh) | CPU P95 (ms/ảnh) | Fit final (s) |
|---|---:|---:|---:|---:|
""" + rows + """

CPU Ryzen 7 5800H, một luồng, 6.000 probe, model nạp sẵn. Tổng thời gian gồm đọc ROI, CompCode và score cho 600 lớp, chưa gồm xếp hạng/UI. Fit final không gồm trích feature/CV. Không cộng các median thành phần để suy ra median tổng.

CI Rank-1 từng run nằm trong bảng CSV, bootstrap theo Palm ID, 1.000 lần, seed=42. Paired CRC − SVM là +0,9000 điểm phần trăm, CI 95% [0,4000; 1,5000]. Xem [bảng paired](tables/paired_rank1_differences.csv) và [lưới CV](tables/tongji_cv_grid_summary.csv).

Classical source SHA-256 khi chạy: 4e324e0b8840cb473dcda4cfcf75c3344b21f64d03360a2efe0bb502ad3b05bc. Mã hiện tại đã được dọn sau benchmark; snapshot gốc và các khóa được giữ nguyên tại máy. Môi trường được lưu trong [experiments/environments](../experiments/environments/README.md).

CompCode là bản Python paper-inspired, chưa xác minh parity MATLAB. Tham khảo [trang tác giả Tongji](https://cslinzhang.github.io/ContactlessPalm/) và [bài báo gốc](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).
"""
    write("reports/TONGJI_BENCHMARK.md", benchmark)

print("Updated README, current status, canonical embedding guide and experiment provenance.")
