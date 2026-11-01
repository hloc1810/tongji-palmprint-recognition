# Đặc tả mô hình và bằng chứng thực nghiệm

## 1. Thuật toán và phạm vi công bố

- **1-NN:** `NearestTemplate` sử dụng bình phương khoảng cách Euclidean trên vector CompCode L2-normalized; mỗi Palm ID lấy khoảng cách đến template gần nhất. Bảng Rank-k được sắp trên score cho *lớp*, không phải nhiều template cùng Palm ID. Không phải k-NN đa số phiếu k=3/5.
- **Linear SVM:** `LinearSVC(loss='squared_hinge', dual=False, multi_class='ovr')`. Điểm nhỏ hơn tốt hơn do `scores=-decision_function`. Điểm không phải xác suất; chọn C trong {0.1,1,10} theo mean CV Rank-1.
- **CRC_RLS:** Dùng nghiệm ridge với từ điển ảnh tất cả lớp, `alpha=A.T @ (A A.T + lambda I)^(-1) y`, rồi tính lỗi tái tạo Euclidean theo mỗi Palm ID. `cho_solve` thực hiện giải hệ mà không nghịch đảo ma trận trực tiếp; test đối chiếu nghiệm ridge tham chiếu. Lambda {0.1,1.35,10}. Lỗi theo lớp không chia cho norm alpha. Eq. (6) của paper Tongji 2017 dùng **bình phương** lỗi đó; bản Python giữ nguyên ranking nhưng trả score là căn bậc hai.
- **CompCode:** 6 Gabor orientation responses và block-wise histogram. Triển khai Python **chưa có đối chứng số học với MATLAB chính thức**, nên gọi `paper-inspired CompCode histogram + CRC_RLS` thay vì khẳng định tái hiện 98,78%.

## 2. Vòng đời thực nghiệm và kiểm soát rò rỉ dữ liệu

`prepare` → `features` (session1) → `cv` (session1 5-fold) → `train` (session1 toàn bộ) cho **cả ba** → `lock` → `evaluate` (session2) → `errors`, `benchmark`, `report`.

- Trước `lock`: lệnh `features` chỉ tạo session1 cache; `cv` và `train` không đọc ảnh test. Manifest được phép kiểm kê ảnh cả hai phiên để xác nhận nhãn/đủ ảnh nhưng không dùng test pixels làm feature hay học tham số. Dữ liệu đầu vào được băm SHA-256 trước cả cache hit; ảnh không khớp manifest bị từ chối.
- `cv`: ghi bảng đủ fold/parameter, `oof_predictions.csv` cho cấu hình được chọn; kiểm tra mỗi ảnh session1 có đúng một dự đoán OOF.
- `train`: xác nhận hash folds, file cấu hình, CV, OOF, mã nguồn và manifest trước khi lưu artifact.
- `lock`: yêu cầu ba model hoàn thành và khóa artifact, source, manifest. Mọi thay đổi sau đó cần workspace nghiên cứu mới; không xóa test để retune.
- `evaluate`: ghi kết quả một lần; nếu code/artifact/config khác lock thì từ chối.
**Lịch sử test:** Chưa có bằng chứng xác nhận pilot HOG được nhắc trong bản tài liệu ban đầu. Benchmark classical và CNN đã quan sát session2 trước khi bổ sung embedding; các so sánh bổ sung là hậu nghiệm. Không dùng test để chọn cấu hình hoặc seed.
- **Hạn chế:** Quy trình CNN theo seed vận hành độc lập, chưa được đăng ký trong khóa chung của ba mô hình cổ điển trước khi xem test. Nếu muốn so sánh CNN như một candidate không thiên lệch, nhóm phải *chốt cấu hình/seeds/commit CNN trước khi chạy lệnh evaluate đầu tiên* và lưu hồ sơ đó vào phụ lục.

## 3. Đặc tả đầu ra phục vụ báo cáo

| Artifact | Nội dung khoa học | Bước sinh |
|---|---|---|
| `data/metadata/manifest.csv` | Session, Palm ID, hash, chất lượng proxy | prepare |
| `data/splits/fold_assignments.csv` | 5 fold class-stratified cho session1 | prepare |
| `outputs/cv/compcode_<m>.csv` | Rank-1/Rank-5/Macro-F1 từng fold và parameter | cv |
| `outputs/cv/compcode_<m>_oof_predictions.csv` | Dự đoán validation độc lập cho mỗi mẫu session1, cấu hình chọn | cv |
| `outputs/cv/compcode_<m>_selection.json` | Best parameter, mean/std, hash CV/OOF/source | cv |
| `artifacts/protocol_lock.json` | Khóa ba model, manifest và nguồn trước test | lock |
| `outputs/runs/<m>/metrics.json` | Rank-1, Rank-5, Precision/Recall/F1 macro, CMC, CI clustered, đo matching | evaluate |
| `outputs/runs/<m>/predictions.csv` | Sample UID, label, dự đoán, true rank, Top-5, điểm khoảng cách/margin/residual | evaluate |
| `outputs/runs/<m>/all_class_scores.npz` | Ma trận N_probe x 600 class score đầy đủ + UID, ID, class order, float64 | evaluate |
| `outputs/runs/<m>/per_class_metrics.csv` | Support, Precision, Recall, F1, đúng/sai cho từng Palm ID | evaluate |
| `outputs/runs/<m>/cmc_full.csv` | Tỷ lệ nhận dạng tại mọi Rank k = 1..600 | evaluate |
| `outputs/runs/<m>/confusion_matrix.npy` | Ma trận 600x600 theo class_order.json | evaluate |
| `outputs/runs/<m>/errors.csv` | Tất cả mẫu Rank-1 sai | evaluate |
| `outputs/runs/<m>/benchmark_cpu.json` | Latency median/P95 có tiền xử lý, feature, matching | benchmark |
| `reports/tables/model_comparison.csv` | Bảng so sánh metric và CI, cảnh báo loại timing | report |
| `reports/tables/paired_rank1_differences.csv` | Chênh lệch Rank-1 của các mô hình trên *cùng probe*, paired palm-bootstrap | report |
| `reports/tables/evidence_verification.csv` | Kiểm tra phép tính từ raw scores và hash | report |
| `reports/tables/cv_selection_summary.csv` | Best CV config mỗi classifier/ablation | report |
| `reports/tables/hardest_palms_<m>.csv` | Các Palm ID recall thấp nhất | report |
| `reports/tables/top_confusions_<m>.csv` | Những cặp nhãn bị lẫn nhiều nhất | report |
| `reports/figures/*` | CMC đầy đủ, Rank-1 CI, so sánh Rank-1/5/F1 | report |

**Bằng chứng cần thu thập riêng từ dữ liệu và thực nghiệm thật:** ảnh minh họa ROI thực, nhận xét thủ công có dẫn chứng cho 10–15 trường hợp sai, đồ thị đúng protocol với MATLAB gốc, phiếu duyệt nguồn dữ liệu từ giảng viên, số liệu benchmark trên Tongji thật, thống kê CNN nhiều seed và slide/report đã hoàn chỉnh. Những mục này bắt buộc được bổ sung từ kết quả chạy thật, không dùng smoke synthetic như kết quả khoa học.

## 4. Kiểm chứng và giới hạn

- Giai đoạn Research correctness: [kết quả audit, kế hoạch và nghiệm thu](RESEARCH_CORRECTNESS.md). Không có các baseline GaussianNB/RandomForest/Raw Pixel được đo trong workspace này; không đưa các số lịch sử thiếu chứng cứ vào bảng chính thức.
- Artifact/freeze và `metrics.json` ghi `model_definition`, `manifest_sha256`, `artifact_sha256`; bảng so sánh giữ tên phương pháp, loại score, phạm vi dữ liệu và các hash. SVM ghi đầy đủ tham số liblinear, CRC ghi rõ không chuẩn hóa hệ số lớp.
- `report` đối chiếu UID/nhãn/thứ tự probe với manifest gốc, hash với frozen artifact, rồi tái tính metric/CI/CMC/Top-5/flags/per-class/errors. Thiếu evidence hoặc có sai khác thì dừng trước khi xuất bảng. Manifest non-strict luôn mang nhãn `NON_STRICT_fixture_or_subset` trong bảng và tiêu đề hình; strict chỉ xác nhận bố cục dữ liệu, không chứng minh nguồn tải hoặc tính độc lập test.
- Cache phiên bản v3 gắn manifest, cấu hình, mã extractor/ROI reader và phiên bản NumPy/OpenCV/scikit-image; cache hit phải đúng UID, shape, dtype float và hữu hạn. Cache cũ được bỏ qua theo signature mới. Artifact cũ thiếu metadata cần quy trình migration có kiểm chứng hoặc workspace thí nghiệm mới; không tự gán metadata và accuracy cũ cho model hiện tại, không xóa run đã test để tune lại.

- Test kỹ thuật: gán nhãn, chia fold, CRC nghiệm ridge, SVM/kNN ranking, output score integrity, fail khi đọc test sớm, code/config/raw tampering, CNN save/load/train synthetic.
- Bootstrap nhóm theo Palm ID; không điều chỉnh sự phụ thuộc hai tay cùng chủ thể vì thiếu ánh xạ person ID.
- Matching time chia theo batch chỉ thể hiện **batch-amortized classifier time**, không phải end-to-end per-ROI latency. Đo `palmprint benchmark --model ...` trên cùng máy.
- SVM 600 lớp và 5-fold ba giá trị C có thể tốn thời gian trên CPU; chuẩn bị máy và ghi wall time, không giảm fold tùy tiện sau khi xem test.
- Không báo cáo các giá trị accuracy từ smoke test là kết quả trên Tongji thật.

## 5. Công thức đối chiếu paper

Lin Zhang et al., *Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach*, Pattern Recognition 69 (2017), pp. 199–212, Section 5, Eqs. (1)–(6), Table 3 và Section 6.3: https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf.

Biểu diễn block của bản Python không được công nhận tương đương MATLAB trước khi thử so sánh một số ROI và intermediate maps/vector. Ràng buộc này phải xuất hiện trong báo cáo.
