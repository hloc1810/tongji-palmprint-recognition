# Kiến trúc repository

## 1. Các tầng mã nguồn

`src/palmprint` là thư viện được cài bằng `pip install -e .`. Các entry point
chỉ điều phối; logic mô hình và đánh giá nằm trong các module chuyên trách.

| Module | Trách nhiệm | Đầu ra chính |
|---|---|---|
| `data/` | Decoder, kiểm tra ROI, ánh xạ Palm ID, manifest, split; Custom ROI | Metadata, fold, ROI |
| `features/` | CompCode, biểu diễn ablation, chọn đặc trưng sparse PalmNet | Vector hoặc CSR descriptor |
| `models/` | 1-NN, SVM, CRC_RLS, ResNet, matching sparse | Điểm cho từng Palm ID |
| `training/` | Chọn cấu hình Session 1, fit final, khóa protocol | Model và metadata freeze |
| `evaluation/` | Rank-k, Macro-F1, CMC, bootstrap, CNN/embedding | Metrics và dự đoán |
| `reporting/` | Xác minh bằng chứng và sinh bảng, hình | Báo cáo dẫn xuất |
| `utils.py` | I/O, hash và thông tin môi trường | JSON có thể truy vết |

Luồng chính là **data → features → model → evaluation → reporting**.
Huấn luyện gọi cùng bộ trích xuất và mô hình; báo cáo kiểm tra số liệu đã lưu,
không huấn luyện lại để cải thiện kết quả.

## 2. Entry point và ứng dụng

| Tác vụ | Entry point |
|---|---|
| M1–M3: prepare, CV, train, lock, evaluate | `palmprint` hoặc `python -m palmprint.cli` |
| M4: CNN và embedding | `palmprint-cnn` hoặc `python -m palmprint.cli_cnn` |
| M5: PCA-Gabor PalmNet | `palmprint-palmnet` hoặc `python scripts/run_palmnet_m5.py` |
| Pilot Custom ROI | `python scripts/run_roi_comparison.py` |
| Minh họa ROI có nhãn | `python scripts/render_roi_visual_review.py` |
| Smoke test synthetic | `python scripts/run_smoke.py` |
| FastAPI + HTML | `python -m uvicorn app.server:app` |
| Streamlit | `python -m streamlit run app/main.py` |

Frontend HTML nằm tại `app/static/index.html`; backend phục vụ cả `/` và URL cũ
`/Tongji_Palmprint_UI.html`. API và Streamlit dùng M1–M3, chưa phục vụ M4/M5/Custom ROI.
Không cần thêm wrapper M5 ở gốc vì CLI và `scripts/` đã bao phủ tác vụ đó.
[Danh mục script](../scripts/README.md) mô tả các chương trình phân tích bổ sung.

## 3. Dữ liệu, model và bằng chứng

| Vị trí | Vai trò và quy tắc |
|---|---|
| `data/raw/session{1,2}/` | Official ROI gốc, chỉ lưu cục bộ |
| `data/Original Images/session{1,2}/` | Ảnh bàn tay gốc, chỉ lưu cục bộ |
| `data/metadata/`, `data/splits/` | Manifest, audit và fold đã lưu |
| `artifacts/` | Model/freeze M1–M4; checkpoint bộ trích xuất M5 trong `pretrained/` |
| `outputs/runs/`, `outputs/cv/` | Bằng chứng nhận dạng và chọn cấu hình |
| `outputs/palmnet_m5/`, `outputs/roi_comparison/` | Hai thực nghiệm bổ sung có giao thức riêng |
| `outputs/*/source_snapshot/` | Mã lịch sử khi benchmark, không phải module đang import |
| `reports/tables/`, `reports/figures/` | Bảng và hình dẫn xuất |
| `.local/` | File tạm, output mới, bản khôi phục và thí nghiệm chưa công bố |

Tên file giống nhau trong `source_snapshot/` và `src/` phục vụ truy vết phiên bản;
không xem chúng là code thừa để xóa hàng loạt. Model, score matrix, manifest,
prediction và protocol đã báo cáo được giữ nguyên. Script trong
`outputs/cnn_experiments/` còn được tham chiếu trong báo cáo CNN; chúng là
hồ sơ lịch sử, không phải entry point đang duy trì.

## 4. Tài liệu và kiểm chứng

[Phương pháp](../docs/METHODOLOGY.md) nêu giả thiết và định nghĩa chỉ số;
[tái lập](../docs/REPRODUCIBILITY.md) nêu lệnh chạy và giới hạn môi trường;
[bảo trì](../docs/MAINTENANCE.md) quy định vị trí file và kiểm tra trước khi commit.
Không đổi các module khoa học đã khóa chỉ để sắp xếp thư mục; thay đổi hành vi
thuật toán cần giao thức và kết quả thực nghiệm mới.
