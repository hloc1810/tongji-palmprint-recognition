# Giai đoạn 1 — Research correctness

> Hồ sơ lịch sử của đợt audit trước benchmark. Các ghi chú ‘chưa chạy’ bên dưới mô tả thời điểm đó. Trạng thái và số liệu hiện tại nằm trong [PROJECT_STATUS.md](PROJECT_STATUS.md) và [benchmark](../reports/TONGJI_BENCHMARK.md).

Ngày kiểm tra: 01/11/2026. Phạm vi: workspace hiện tại, không có thư mục `.git`, nên không xác định được commit hoặc lịch sử các bản mã cũ.

## 1. Hiện trạng dự án

| Thành phần | Đã triển khai | Chưa được chứng minh tại thời điểm bắt đầu audit |
|---|---|---|
| `data/`, `configs/` | Hai session, quy tắc 10 ảnh/Palm ID, manifest/hash, 5 fold session1 | Tính hợp lệ của toàn bộ ảnh thực tế |
| `src/palmprint/data/` | Đọc Official ROI 128×128, kiểm kê nhãn/chất lượng, phát hiện trùng hash | Không có bộ trích ROI từ ảnh toàn bàn tay |
| `features/` | CompCode 6 hướng, histogram block 600 chiều, global/HOG ablation, cache theo signature | Parity với MATLAB tác giả |
| `models/`, `training/` | 1-NN, sklearn LinearSVC, collaborative ridge; CV/freeze/lock/evaluate | Chưa có artifact huấn luyện hoặc kết quả Tongji trong workspace |
| `evaluation/`, `reporting/` | Rank-k, macro metrics, CMC, confusion, bootstrap, latency, bảng từ score evidence | Chưa có benchmark 600 lớp đã chạy |
| CNN | ResNet18, chọn lịch trên session1 rồi refit, checkpoint theo seed | Pretrained multi-seed Tongji; khóa CNN chung với classical |
| `app/`, `notebooks/`, `reports/` | Demo một model/lượt, notebook đọc kết quả, đề cương báo cáo/slide | Báo cáo hoàn chỉnh và đánh giá giao diện |
| `tests/`, `.github/` | Unit/integration synthetic, CI Python 3.11/3.12 | Kết quả chạy lại trên bản sửa hiện tại |

Luồng chính: Official ROI → manifest/folds → feature session1 → CV → fit ba model → lock → feature/session2 evaluation → kiểm chứng evidence → report/demo. `experiments/registry.yaml` là thiết kế thí nghiệm, không phải kết quả đo.

## 2. Đối chiếu ba yêu cầu P0

1. **CI / convexity defects:** không tìm thấy `convexityDefects` hay vòng lặp `defects[i, 0]` trong bản này. Không thêm bộ trích ROI không được kiểm chứng vào pipeline Official ROI. Với bản cũ, `reshape(-1, 4)` xử lý cả `(N,1,4)` và `(N,4)`; cần kiểm tra `None` trước và chia depth cố định điểm cho 256. Sửa đó chỉ thay cách đọc cấu trúc khi vẫn giữ nguyên chỉ số contour và ngưỡng. Không thể tuyên bố đã sửa một file không tồn tại.
2. **Benchmark:** không có `generate_cross_domain_evidence.py`, GaussianNB, RandomForest hay Raw Pixel baseline trong mã hiện tại; không tìm thấy các số 10,10%, 28,47%, 76,55%, 98,05%. Chọn phương án **không đưa các hàng thiếu bằng chứng vào bảng chính thức**. Không tạo thêm thí nghiệm session2 nhằm hợp thức hóa số cũ. Nhãn và thứ tự phải đi từ manifest, cache phải gắn code/config/dữ liệu, report phải đối chiếu score với nhãn gốc.
3. **Toán học:** CRC hiện tính residual Euclidean không chuẩn hóa. Eq. (6) của Tongji tính residual bình phương không chuẩn hóa; căn bậc hai giữ nguyên ranking nhưng giá trị score khác. Không tự chuyển sang chia `||alpha_c||`. SVM hiện là `LinearSVC` squared-hinge OvR có intercept; không có Scratch SVM analytic để đổi tên. Nếu khôi phục nghiệm `W=(XᵀX+λI)⁻¹XᵀT` không bias từ bản cũ, phải gọi **multi-output ridge classifier, no intercept**, không gọi LinearSVC hoặc LS-SVM chuẩn. Không kế thừa accuracy cũ sang model hiện tại.

## 3. Kế hoạch triển khai và tiêu chí nghiệm thu

| Thứ tự | Công việc | Tiêu chí |
|---|---|---|
| P0.1 | Cài môi trường tách biệt, chạy baseline và regression Python 3.11/3.12 | Có log pytest, phiên bản, dependency và skip thực tế; CI giữ matrix hai bản |
| P0.2 | Kiểm kê toàn bộ dữ liệu, xem ROI mẫu và chạy feature trên ROI thật | 600 Palm ID, 6.000 ảnh/session, 128×128 uint8; không đổi thuật toán ROI |
| P0.3 | Siết cache, nhãn/thứ tự manifest và chứng cứ report | Cache sai shape/thứ tự bị chặn; nhãn giả, metric/CI/flags sai bị chặn; non-strict được ghi rõ |
| P0.4 | Ghi biến thể model vào artifact/metrics/bảng; bổ sung đối chứng toán học | CRC khớp nghiệm ridge trực tiếp và khác normalized residual; SVM khớp sklearn có bias |
| P0.5 | Chạy lại toàn bộ test, smoke, lưu hồ sơ | Không đưa synthetic accuracy hoặc accuracy cũ thành kết quả Tongji |

## 4. Cơ sở khoa học

- [Trang tác giả Tongji](https://cslinzhang.github.io/ContactlessPalm/): 300 người, 600 lòng bàn tay, 12.000 ảnh; hai session cùng tên ứng với cùng lòng bàn tay.
- [Zhang et al., Pattern Recognition 69 (2017), §5, Eq. (4)–(6), Table 3](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf): ridge toàn dictionary và residual bình phương không chuẩn hóa. Bản Python dùng Cholesky trong không gian feature; đây là cùng nghiệm ridge với λ>0, không phải bằng chứng parity toàn pipeline.
- [Tài liệu LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html): squared-hinge, OvR; liblinear phạt cả trọng số synthetic intercept. Phân biệt loss này với bình phương lỗi nhãn trong ridge regression.

Chỉ dùng số thực nghiệm có cấu hình và evidence tương ứng. Kiểm tra tính đúng phần mềm không đồng nghĩa xác nhận accuracy, khả năng tổng quát hóa hoặc tái hiện MATLAB.

## 5. Kết quả thực hiện và bằng chứng

| Kiểm tra | Python 3.11.15 | Python 3.12.13 |
|---|---|---|
| Baseline trước sửa code | 26 passed, 7 skipped (thiếu torch) | 26 passed, 7 skipped (thiếu torch) |
| Toàn bộ test sau sửa, có torch CPU | **51 passed**, không skip, 23,73 giây | **51 passed**, không skip, 26,23 giây |
| Smoke classical end-to-end | Đạt | Đạt |
| Compile `src`, `app`, `scripts` | Đạt | Đạt |

Đã bổ sung 18 ca kiểm thử, bao gồm: đối chứng liblinear có intercept với ridge không intercept; residual CRC trực tiếp, batch invariance và ví dụ normalized residual đổi ranking; manifest bị xáo thứ tự; cache sai số hàng/cột/UID; giả mạo metrics, CI, flags, Top-5, class order, hash manifest, model definition, score kind và artifact. Nhánh CNN train/evaluate synthetic cũng được kiểm tra xuất report với metadata mới. Ruff kiểm tra import và lỗi F trên các file Python đã sửa: đạt.

**Dữ liệu thực tế:** strict audit đã đọc/kiểm kê toàn bộ 12.000 ROI: 600 Palm ID × 10 ảnh × 2 session, 128×128 uint8, không có nhóm trùng SHA-256 (cả nội bộ và giữa hai session). Có 5 fold session1, seed=42. Manifest SHA-256: `99bbbe45a4807d48801a28a6aadfec2efee46a3f4c4fe3d3af8e2337241c8ae0`.

Kiểm tra feature thật trên Palm ID cố định 1, 150, 300, 450, 600 thuộc session1: đủ các mã 0…5, vector 600 chiều, norm L2 từ 0,99999994 đến 1. Đã xem trực quan `session1/00001.bmp` và `session2/06000.bmp`; đây là ROI đường vân lòng bàn tay, không phải ảnh toàn bàn tay. Kiểm tra này không xác nhận chất lượng căn chỉnh của mọi ảnh, nguồn tải chính thức hoặc parity MATLAB. Không dùng feature session2 để chọn cấu hình hay đo accuracy trong giai đoạn này.

**Thay đổi đã áp dụng:**

- SVM khai báo tường minh các mặc định đang dùng; không đổi loss, solver hoặc phương pháp. CRC giữ nguyên tính toán không chuẩn hóa. Artifact/freeze/metrics/bảng lưu tên biến thể và công thức score, SVM lưu tham số đã fit.
- Cache v3 gắn thêm mã ROI reader và phiên bản thư viện; kiểm tra đúng shape, dtype, UID và giá trị hữu hạn khi đọc cache.
- Report xác minh toàn bộ run trước khi xuất bảng: so manifest gốc, hash artifact/freeze và model definition; tái tính metric, CI, CMC, Top-5, đúng/sai và errors. Non-strict fixture/subset được ghi rõ trong bảng và tiêu đề hình.
- CI giữ Python 3.11/3.12, bổ sung smoke test, JUnit, dependency snapshot và upload evidence. Chưa chạy GitHub Actions từ workspace này; xác minh hai phiên bản ở trên là kết quả local Windows.

**File bằng chứng cục bộ:** `outputs/research_correctness/roi_audit.json`, `python311-pytest.xml`, `python312-pytest.xml`, hai log `*-pytest.txt`, hai log `*-smoke.txt`, hai snapshot `*-dependencies.txt` và `validation_summary.json`. Dữ liệu/đường dẫn/hash từng ảnh nằm trong `data/metadata/` và `data/splits/`, được gitignore. Các bằng chứng local không nằm trong gói source hoặc được tự sinh trong CI; không hardcode số liệu audit vào thuật toán.

Tái lập sau khi cài package và extras tương ứng:

```powershell
python -m pytest -q -ra --junitxml outputs/research_correctness/pytest.xml
python scripts/run_smoke.py
python scripts/audit_research_correctness.py --data data/raw
```

Lệnh audit thực hiện `prepare` strict; dùng trước khi khóa thí nghiệm. Nếu workspace đã lock, giữ nguyên hồ sơ và chạy audit trong workspace mới có `configs/` thay vì xóa lock.

**Phần còn lại ngoài nghiệm thu này:** chưa có số Rank-1/5 Tongji mới, chưa có GaussianNB/RandomForest/Raw Pixel benchmark có huấn luyện, chưa có parity MATLAB, chưa chạy pretrained CNN đa seed trên Tongji. Các hàng accuracy cũ không xuất hiện trong bảng kết quả chính thức; không có cơ sở dùng lại 98,05% cho SVM hiện tại. Vấn đề `defects` ở bản cũ được đánh dấu không áp dụng cho snapshot này, không tuyên bố đã sửa hoặc kiểm thử bộ trích ROI không tồn tại.
