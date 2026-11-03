# Hướng dẫn kiểm tra và tái lập

## 1. Ba mức kiểm chứng

| Mức | Yêu cầu | Ý nghĩa |
|---|---|---|
| Kiểm tra kết quả lưu | Repository, dependency liên quan | Đối chiếu metric, prediction và protocol |
| Kiểm thử phần mềm | Dependency test; ảnh synthetic do test tạo | Kiểm tra hình học, mô hình, khóa và API |
| Chạy lại Tongji | Dataset, môi trường thích hợp, workspace mới | Tính lại pipeline trên dữ liệu thật |

GitHub không chứa 24.000 ảnh của hai bộ Original/Official, gallery/cache PalmNet
lớn hoặc mọi score matrix M5. Clone và đọc bảng chưa phải tái lập benchmark.
Các artifact có sẵn hỗ trợ inference và truy vết, không thay cho chạy lại.

## 2. Môi trường

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[test,api,demo]"
python -m palmprint.cli --help
```

Python 3.11/3.12 là hai phiên bản trong CI. Linux/macOS kích hoạt môi trường
bằng `source .venv/bin/activate`. Cài thêm `.[cnn]` cho ResNet18, `.[palmnet]`
cho M5. M5 còn cần Python port của tác giả tại commit chỉ định trong
[báo cáo M5](../reports/PALMNET_M5.md).

DirectML dùng môi trường Windows riêng. Các [snapshot dependency](../experiments/environments/README.md)
ghi lại môi trường đã chạy, không phải lockfile đa nền tảng. Thay thư viện/backend
có thể thay đổi timing, sai số hoặc kết quả; ghi phiên bản của lần chạy mới.

## 3. Dataset và hiện vật lịch sử

Theo [hướng dẫn dataset](../data/README.md), đặt Official ROI trong `data/raw/`,
ảnh gốc trong `data/Original Images/`, mỗi bộ 6.000 ảnh/session. Không đổi tên,
trộn phiên hoặc resize Original thành Official ROI.

Metadata đi kèm có thể chứa đường dẫn máy cũ và checksum theo byte của CSV.
Không sửa metadata/khóa cũ để vượt verifier; chuẩn bị manifest mới trong
workspace mới từ ảnh dữ liệu gốc. Git chuẩn hóa xuống dòng ở một số file lịch
sử có thể làm checksum sau clone khác lúc benchmark; cần phân biệt với ảnh bị
đổi. CSV/JSON pilot ROI được cấu hình giữ nguyên byte trong `.gitattributes`.

## 4. Kiểm thử không cần dataset

```powershell
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q -ra -p no:cacheprovider --basetemp .local/tests
python scripts/run_smoke.py
```

Test tạo ảnh synthetic trong thư mục tạm. Một số test cần dependency tùy chọn
có thể skip; đọc tổng kết pytest trước khi nói đã kiểm thử đầy đủ. Smoke chạy
prepare→CV→train→lock→evaluate→report trên ảnh synthetic, không đo accuracy Tongji.

## 5. M1–M3 trong workspace mới

Ví dụ PowerShell dùng mã đã cài nhưng ghi kết quả mới vào `.local/`. Chọn tên
workspace chưa có model/protocol, không xóa khóa của benchmark đã công bố.

```powershell
$trial = Join-Path (Get-Location) '.local/experiments/classical'
New-Item -ItemType Directory -Force -Path $trial | Out-Null
Copy-Item -LiteralPath configs -Destination $trial -Recurse

python -m palmprint.cli --workspace $trial prepare --data ./data/raw
foreach ($method in @('knn', 'svm', 'crc')) {
    python -m palmprint.cli --workspace $trial cv --model $method
    python -m palmprint.cli --workspace $trial train --model $method
}
python -m palmprint.cli --workspace $trial lock
foreach ($method in @('knn', 'svm', 'crc')) {
    python -m palmprint.cli --workspace $trial evaluate --model $method
}
python -m palmprint.cli --workspace $trial report
```

Kiểm tra exit code sau mỗi lệnh; xử lý lỗi trước bước tiếp theo. Sau khóa,
evaluation từ chối ghi đè kết quả. Lựa chọn cấu hình mới phải dùng Session 1
trong nghiên cứu mới, không sửa tham số theo kết quả Session 2.

## 6. Pilot ROI

```powershell
python scripts/run_roi_comparison.py --output .local/experiments/roi_repeat
```

Mặc định chọn 10 ID bằng seed 42, ghép đúng 200 cặp Original/Official.
Có thể cấu hình `--original-root`, `--official-root`, `--seed`, `--output`.
Đổi seed/thuật toán sau khi xem kết quả là một nghiên cứu thăm dò mới.

Output có manifest/hash, protocol, summary, bảng so sánh, hai bảng prediction,
trạng thái crop, landmark, ROI và contact sheet. Thời gian ROI không gồm decode,
I/O, CompCode và 1-NN. `scripts/render_roi_visual_review.py` chỉ vẽ các ví dụ
đã gán nhãn của pilot lưu tại `outputs/roi_comparison/`; không chấm đúng/sai
tự động cho output hoặc seed bất kỳ.

## 7. CNN và PalmNet

M4 dùng `palmprint-cnn --workspace <workspace-mới>` và `configs/resnet18.yaml`.
Xem [CNN](../reports/CNN_EXPERIMENTS.md) để phân biệt logits, embedding hậu nghiệm
và kiểm chứng sai số float32. M5 dùng `python scripts/run_palmnet_m5.py --help`;
thứ tự train→train-classifiers→evaluate→evaluate-classifiers và chi phí nhiều
GiB được mô tả trong [M5](../reports/PALMNET_M5.md).

Các runner trong `outputs/cnn_experiments/` là hồ sơ lịch sử có giả định riêng
về đường dẫn và môi trường. Dùng CLI hiện tại cho workspace mới; chỉ dùng script
lịch sử khi chủ động tái dựng đúng môi trường đó.

## 8. Ghi nhận một kết quả mới

Lưu commit, SHA256/cấu hình, Python/dependency, dataset/manifest, session split,
seed, cách chọn cấu hình, mẫu số và phạm vi timing. Bảng phải dẫn đến prediction
và metric tương ứng. Không sửa bằng chứng cũ để khớp với một bảng hoặc mã mới.
