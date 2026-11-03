# Tongji Contactless Palmprint Recognition

Đồ án môn **Nhận dạng mẫu**: nhận dạng lòng bàn tay không tiếp xúc theo giao thức
**closed-set 1:N**, trên Tongji Contactless Palmprint Dataset. Nhóm 01, đề tài 09.
Mỗi bàn tay là một Palm ID; hai bàn tay của cùng người là hai lớp riêng.

## 1. Bài toán và phạm vi

Benchmark chính sử dụng **Official ROI grayscale 128×128**: 600 Palm ID, 12.000 ảnh,
hai phiên chụp. Session 1 có 6.000 ảnh dùng cho phát triển mô hình và gallery;
Session 2 có 6.000 probe dùng để đánh giá chéo phiên. Các thực nghiệm có lịch sử
quan sát Session 2 được ghi rõ trong từng báo cáo.

```text
Official ROI → biểu diễn đặc trưng → điểm cho từng Palm ID → xếp hạng 600 lớp
Ảnh gốc → Custom ROI 128×128 → CompCode → 1-NN              [pilot 10 lớp]
```

Đóng góp triển khai gồm CompCode histogram không gian, 1-NN/CRC_RLS, so sánh
ResNet18 và PalmNet, cùng pilot ROI tự triển khai theo hình học Tongji 2017.
Không khẳng định tương đương từng pixel với mã MATLAB của tác giả.

## 2. Kết quả được lưu trong repository

### Benchmark 600 Palm ID

| Pipeline | Biểu diễn và bộ nhận dạng | Rank-1 (%) | Rank-5 (%) |
|---|---|---:|---:|
| M1 | CompCode 600-D + 1-NN Euclidean | 94,4333 | 96,9667 |
| M2 | CompCode 600-D + Linear SVM | 97,3667 | 98,5167 |
| M3 | CompCode 600-D + CRC_RLS | 98,2667 | 99,0833 |
| M4-E1 | ResNet18, logits; trung bình 3 seed | 64,3611 ± 1,0409 | 81,9111 ± 0,6545 |
| M4-E2 | ResNet18, nearest cosine; hậu nghiệm, 3 seed | 73,7444 ± 0,6241 | 87,5000 ± 0,4368 |
| M5 | PCA-Gabor PalmNet + nearest template | 99,9833 | 100,0000 |

Mỗi run có 6.000 probe. Dấu ± là **độ lệch chuẩn mẫu giữa 3 seed**, không phải CI.
M5 đúng 5.999/6.000 probe; là thực nghiệm độc lập với descriptor sparse 12.288.000-D.
Các pipeline khác nhau về cách học và chọn cấu hình; không quy toàn bộ chênh lệch
cho riêng bộ phân loại. Chi tiết: [M1–M4](reports/TONGJI_BENCHMARK.md),
[CNN](reports/CNN_EXPERIMENTS.md), [M5](reports/PALMNET_M5.md).

### Pilot ROI: cùng 10 Palm ID, 200 ảnh

| Chỉ số | Official ROI | Custom ROI |
|---|---:|---:|
| Ảnh đầu vào | 200 | 200 |
| Gallery dùng được / probe dùng được | 100 / 100 | 99 / 100 |
| Crop được chấp nhận tự động | Không áp dụng | 199/200, tức 99,5% |
| Rank-1, mẫu số 100 probe | 99% | 98% |
| Thời gian riêng thuật toán ROI trung bình | Không áp dụng | 2,39 ms/ảnh |

Tỷ lệ chấp nhận tự động **không phải tỷ lệ crop đúng giải phẫu**; đã xác nhận bốn
ví dụ chọn sai kẽ ngón. Không so trực tiếp pilot này với benchmark 600 lớp.
Xem [phương pháp, bảng kết quả và lỗi ROI](reports/ROI_COMPARISON.md).

## 3. Cài đặt và chạy demo

Python **3.11 hoặc 3.12**; chạy các lệnh từ thư mục gốc repository.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[test,api,demo]"

# FastAPI và frontend HTML
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

# Hoặc demo Streamlit
python -m streamlit run app/main.py
```

Mở `http://127.0.0.1:8000` khi dùng FastAPI. Demo nhận **Official ROI 128×128**
và nạp các model CompCode M1–M3 trong `artifacts/`. M4, M5 và Custom ROI chưa
được nối vào API nhận dạng. Tải dữ liệu theo [hướng dẫn dataset](data/README.md)
để sử dụng ảnh mẫu; dataset không được đóng gói trong Git.

CNN/PalmNet cần dependency riêng: `python -m pip install -e ".[cnn,palmnet]"`.
PalmNet còn cần mã tác giả tại commit chỉ định trong [báo cáo M5](reports/PALMNET_M5.md).

## 4. Kiểm tra và tái lập

```powershell
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q -p no:cacheprovider --basetemp .local/tests
python scripts/run_smoke.py

# Pilot ROI: cần đủ ảnh gốc và Official ROI; dùng output mới
python scripts/run_roi_comparison.py --output .local/experiments/roi_repeat
```

Smoke test dùng ảnh **synthetic**, chỉ kiểm tra phần mềm; không phải kết quả Tongji.
Các test cần dependency tùy chọn có thể được bỏ qua nếu môi trường chưa cài chúng.
Không chạy lại `prepare`, huấn luyện hoặc đánh giá vào workspace đã khóa.
[Hướng dẫn tái lập](docs/REPRODUCIBILITY.md) mô tả workspace mới, thứ tự lệnh và
giới hạn của các file có trên GitHub.

## 5. Cấu trúc và tài liệu

| Thư mục | Nội dung |
|---|---|
| `src/palmprint/` | Mã thư viện: dữ liệu, đặc trưng, mô hình, huấn luyện, đánh giá, báo cáo |
| `app/` | Streamlit, FastAPI và `static/index.html` |
| `scripts/` | Entry point chạy thí nghiệm và vẽ hình đang được duy trì |
| `configs/` | Cấu hình phương pháp và giao thức |
| `tests/` | Unit test và integration test bằng dữ liệu synthetic |
| `docs/` | Phương pháp, tái lập, bảo trì và đề bài gốc |
| `reports/` | Báo cáo khoa học, bảng và hình kết quả |
| `outputs/` | Bằng chứng thực nghiệm; snapshot cũ có chủ đích để truy vết |
| `artifacts/`, `data/` | Model, metadata; dữ liệu và cache lớn được giữ cục bộ |
| `.local/` | Kiểm thử tạm, lần chạy mới và kho khôi phục cục bộ; không đưa lên Git |

- [Mục lục tài liệu](docs/README.md) · [Mô tả dữ liệu bài nộp](DATASET.md) · [Phương pháp và tiêu chí đánh giá](docs/METHODOLOGY.md).
- [Kiến trúc repository](reports/REPOSITORY_GUIDE.md) · [Danh mục báo cáo](reports/README.md).
- [Hồ sơ thực nghiệm](experiments/README.md) · [Môi trường đã chạy](experiments/environments/README.md).
- [Quy tắc bảo trì](docs/MAINTENANCE.md) · [Lịch sử dọn repository](docs/REPOSITORY_CLEANUP.md).
- [Đề bài môn học](docs/references/project_assignment.pdf).

## 6. Nguồn tham khảo

1. Zhang et al. (2017), *Towards contactless palmprint recognition: A novel device,
   a new benchmark, and a collaborative representation based identification approach*,
   Pattern Recognition 69, 199–212. [DOI](https://doi.org/10.1016/j.patcog.2017.04.016),
   [dataset và mã tác giả](https://cslinzhang.github.io/ContactlessPalm/).
2. Genovese et al. (2019), *PalmNet: Gabor-PCA Convolutional Networks for Touchless
   Palmprint Recognition*, IEEE TIFS. [Bài báo](https://air.unimi.it/handle/2434/637849),
   [Python port được dùng](https://github.com/AngeloUNIMI/PalmNet/tree/f0a12e6d5f585c7eb0bf5c6804a98bd1d6587e22).

Các số liệu trong README là kết quả của repository, không phải accuracy được chép
từ bài báo. Giấy phép của mã PalmNet bên ngoài áp dụng riêng cho nguồn đó;
repository này chưa công bố giấy phép riêng cho toàn bộ mã dự án.
