# KẾ HOẠCH THỰC HIỆN ĐỒ ÁN CUỐI KỲ MÔN NHẬN DẠNG MẪU — NHÓM 01

**Đề tài:** So sánh các thuật toán nhận dạng lòng bàn tay không tiếp xúc sử dụng đặc trưng Competitive Code và học sâu trên bộ dữ liệu Tongji
**English:** *Cross-Session Contactless Palmprint Identification: Competitive-Code Features, Classical Classifiers, and a CNN Comparison*
**Repository:** <https://github.com/hloc1810/tongji-palmprint-recognition>
**Mục tiêu:** Đáp ứng các tiêu chí đánh giá của giảng viên (9,0 điểm nội dung chính và tối đa 1,0 điểm thưởng) thông qua quy trình thực nghiệm có thể kiểm chứng, mã nguồn tái lập và sản phẩm minh họa.

**Phạm vi nghiên cứu:** Ba thuật toán nhận dạng (k-NN, SVM và CRC_RLS) sử dụng chung đặc trưng CompCode histogram; ResNet18 là nhánh CNN độc lập. Các mô hình được đánh giá theo giao thức cross-session thống nhất, kèm kiểm tra khả năng tái lập, tính so sánh và phân tích sai số.

## 0. TỔNG QUAN KẾ HOẠCH

| Hạng mục | Phương án | Cơ sở lựa chọn |
|---|---|---|
| Bài toán | **Closed-set identification 1:N, 600 Palm ID** | Đúng nhãn Tongji; không nhầm với 600 người hoặc verification 1:1 |
| Ảnh đầu vào | **Official ROI** của Tongji | Kiểm soát tác động từ sự khác biệt trong quy trình trích xuất ROI |
| Biểu diễn chung | **Block-wise histogram của Competitive Code (CompCode)** | Đặc trưng định hướng chuyên biệt cho đường vân, dựa trên paper Tongji |
| M1 | **1-NN / nearest-template** | Thuật toán nền tảng, Ch. 4 giáo trình |
| M2 | **Linear SVM** | Thuật toán nền tảng, Ch. 5 giáo trình |
| M3 | **CRC_RLS** | Collaborative Representation classifier, là hướng chuyên biệt của bài báo |
| M4 | **ResNet18 pretrained + transfer learning** | CNN bổ sung, sử dụng ROI làm đầu vào và học đặc trưng tự động |
| Chọn tham số | **5-fold, mỗi fold 8 train + 2 val / Palm ID trong session1** | Hạn chế phụ thuộc vào một lần tách ngẫu nhiên |
| Final | **Session1 6.000 train/gallery → Session2 6.000 test/probe** | Giao thức cross-session của Tongji |
| Chỉ số chính | Rank-1, Rank-5, Macro-F1, CMC, confusion matrix, thời gian | Đánh giá hiệu năng nhận dạng và chi phí suy luận |
| Phân tích bổ sung | Feature control, spatial ablation, failure analysis, paired CI | Xác định yếu tố ảnh hưởng đến kết quả và mức độ khác biệt giữa các mô hình |

**Mức độ ưu tiên:** P0 gồm chuẩn bị dữ liệu, M1–M3, đánh giá và demo; P1 gồm M4 ResNet18 và các nghiên cứu ablation; P2 gồm các khảo sát bổ sung. M4 thuộc phạm vi triển khai của đồ án và cần phân bổ tài nguyên huấn luyện phù hợp.

---

## 1. ĐỐI CHIẾU YÊU CẦU GIẢNG VIÊN VÀ GIÁO TRÌNH

**Đề giảng viên:** *BÀI TẬP LỚN CUỐI KỲ — MÔN NHẬN DẠNG MẪU*, đề tài 9 cho Nhóm 01. Nhóm 2–3 sinh viên; **nguồn dataset phải được GV duyệt**; báo cáo khoảng 20–30 trang, mã nguồn, slide, bảng phân công; nộp lên GitHub và gửi link theo yêu cầu lớp.

| Tiêu chí | Điểm | Bằng chứng phải nộp |
|---|---:|---|
| Xác định bài toán và dữ liệu | 1,5 | Nêu 600 Palm ID, nguồn Tongji, lý do chọn, EDA, duyệt nguồn |
| Tiền xử lý và biểu diễn | 1,5 | Official ROI, CompCode, kiểm tra dữ liệu, split, chuẩn hóa và giải thích |
| Xây dựng ≥2 mô hình, ≥1 thuộc môn học | 3,5 | M1–M3 + M4; công thức, tham số, code, artifact, log |
| Đánh giá và so sánh | 2,0 | Rank-1/5, Macro-F1, confusion, CMC, bảng kết quả và phân tích lỗi |
| Kiểm tra thực nghiệm/demo | 0,5 | Tải ảnh ROI, dự đoán Palm ID, Top-5 và ảnh đối chiếu bằng model thật |
| Thưởng sáng tạo | +0,5 | CRC_RLS/CompCode nghiên cứu có chiều sâu và CNN có thực nghiệm kiểm chứng |
| Thưởng sản phẩm | +0,5 | Demo trực quan, chạy ổn định, trình bày thuyết phục |
| **Tổng thang điểm** | **10,0** | Điểm thực tế do giảng viên chấm theo mức độ đáp ứng các tiêu chí |

**Giáo trình:** R. O. Duda, P. E. Hart, D. G. Stork, *Pattern Classification*, 2nd ed. Chương 4 (nearest-neighbor), Chương 5 (linear discriminant/SVM), Chương 6 (neural networks), Chương 9 (đánh giá và so sánh classifiers). CompCode/CRC_RLS/ResNet18 phải dẫn nguồn riêng; **không nhận chúng là thuật toán nguyên bản do giáo trình này đề xuất**.

**Câu hỏi nghiên cứu:**

- **RQ1:** Khi **giữ cố định đặc trưng CompCode**, 1-NN, SVM và CRC_RLS khác nhau thế nào về Rank-1/5, Macro-F1 và tốc độ?
- **RQ2:** Histogram **theo block** có mang lại lợi ích thực nghiệm so với histogram **toàn ROI** không? Điều đó đổi chiều vector, nên đây là so sánh *hai biểu diễn*, không quy mọi chênh lệch chỉ cho thông tin không gian.
- **RQ3:** CNN ResNet18 học trực tiếp trên ROI đạt hiệu quả cross-session ra sao so với các pipeline thủ công? Mỗi bên khác cả feature và classifier, vì vậy chỉ kết luận ở **cấp pipeline**.
- **RQ4:** Những nhóm ảnh nào thất bại, có liên quan thế nào đến blur, brightness, độ lệch ROI và khác biệt session? Mô hình nào đánh đổi Accuracy–Latency tốt nhất?

**Tiêu chí nghiệm thu học thuật:** Có kết quả thực nghiệm, phân tích và bằng chứng tái lập cho RQ1–RQ4. Không đặt trước một ngưỡng độ chính xác hoặc thứ hạng mô hình như điều kiện đánh giá.

---

## 2. DATASET, GIAO THỨC VÀ GIỚI HẠN ĐÁNH GIÁ

### 2.1. Tongji và quy tắc nhãn

Nguồn gốc: <https://cslinzhang.github.io/ContactlessPalm/>; bài báo Zhang et al., *Pattern Recognition* **69 (2017), 199–212**. Dữ liệu theo công bố gồm **300 người / 600 lòng bàn tay / 12.000 ảnh**, hai phiên, mỗi phiên **6.000 ảnh**; mỗi Palm ID có **10 ảnh mỗi phiên**. Official ROI là ảnh đã do **tác giả dataset trích**, không phải kết quả phát triển ROI extractor của nhóm.

- Với tên file là số nguyên `n ∈ [1,6000]`: `palm_id = (n-1)//10 + 1` trong **mỗi phiên**.
- Tên file giống nhau ở hai session → **cùng Palm ID**, nhưng là **lần chụp khác**.
- Ảnh Original và Official ROI trùng `session/stem` → **hai biểu diễn của cùng mẫu**; được liên kết cùng `sample_uid` và chỉ tính là một mẫu thu nhận.
- Không tự suy ra tay trái/phải hoặc Person ID từ thứ tự Palm ID nếu thiếu metadata được xác nhận.
- Chỉ sử dụng dữ liệu sinh trắc học theo điều khoản nguồn; **không commit 12.000 ảnh vào repo công khai**.

**D0 — phê duyệt nguồn:** Cung cấp đường dẫn dataset và bài báo gốc cho giảng viên; lưu thông tin xác nhận. Trạng thái phê duyệt được ghi theo bằng chứng thực tế.

### 2.2. Audit và EDA tối thiểu

Script `validate_dataset.py` phải xuất: `manifest.csv` (sample_uid, palm_id, session, path, shape, dtype, SHA256), thống kê số ảnh mỗi Palm ID/session, ảnh lỗi, file trùng hash/pixel/near-duplicate, phân bố intensity/contrast/blur-proxy và hình ảnh đối chiếu 2 session của cùng Palm ID. Ghi rõ blur-proxy chỉ là chỉ báo chẩn đoán, không phải chất lượng sinh trắc có nhãn.

**D1 — tiêu chí nghiệm thu dữ liệu:** Đối soát 600 Palm ID và 10 ảnh/ID/session; ghi nhận các ngoại lệ nếu có; kiểm tra khả năng đọc tệp, nhãn và mẫu trùng lặp. Thống kê trên dữ liệu sử dụng thực tế, phân biệt với số liệu do tác giả công bố.

### 2.3. Phát triển trên session1, kiểm tra trên session2

| Vùng dữ liệu | Số ảnh nếu đủ | Công dụng |
|---|---:|---|
| Session1 — CV training (mỗi fold) | 4.800 | Train classifier, build gallery, fit biến đổi *học từ dữ liệu* |
| Session1 — CV validation (mỗi fold) | 1.200 | Chọn feature/siêu tham số, đánh giá nội bộ |
| Session1 — Final train/gallery | 6.000 | Huấn luyện lại M1–M3 theo cấu hình đã chọn và M4 theo lịch huấn luyện cố định từ giai đoạn phát triển |
| Session2 — Final probe/test | 6.000 | Đánh giá cross-session, **không tối ưu siêu tham số** |

**5-fold cụ thể:** với mỗi Palm ID, sắp 10 ảnh session1 theo stem, chia thành 5 phần 2 ảnh; mỗi fold chọn 1 phần val, 4 phần train. Seed cố định 42; mọi classifier **dùng cùng fold ID**. Mỗi ảnh session1 xuất hiện làm validation đúng một fold. Lưu `fold_assignments.csv` và hash split.

- CV trong cùng phiên chỉ là **ước lượng để chọn cấu hình**, có thể cao hơn kết quả cross-session. Không gọi CV này là phép đánh giá cross-session.
- Không fit scaler, PCA, histogram bảng mã đặc trưng (codebook) có học, lựa chọn tham số Gabor hay CNN bằng session2.
- Sau khi chọn tham số trên session1: M1–M3 huấn luyện lại hoặc lập gallery trên **toàn bộ session1**. ResNet18 xác định lịch epoch từ validation nội bộ, sau đó huấn luyện lại trên toàn bộ session1 theo lịch cố định. Session2 không được sử dụng để chọn checkpoint.
- Theo paper gốc, tham số được lựa chọn trên **tập bổ sung 2.000 ảnh/100 palms**. Giao thức này khác với CV nội bộ của nhóm; khi dẫn kết quả từ paper cần ghi rõ khác biệt về phân chia dữ liệu, phương pháp và phần cứng.

**Lịch sử test:** Chưa có bằng chứng xác nhận pilot HOG được nhắc trong bản tài liệu ban đầu. Benchmark classical và CNN đã quan sát session2 trước khi bổ sung embedding; các so sánh bổ sung là hậu nghiệm. Không dùng test để chọn cấu hình hoặc seed.

**S0 — nghiệm thu:** unit tests cho không giao nhau `sample_uid`, đúng số ảnh, đủ 600 ID, train/dev chỉ session1, final test chỉ session2. Với bài toán closed-set, **cùng Palm ID tồn tại ở train và test là đúng**, cùng tệp ảnh thì không.

---

## 3. FEATURE CHUNG: COMPETITIVE CODE + BLOCK-WISE HISTOGRAM

### 3.1. Đúng tên và đúng paper

Pipeline của paper: ROI → **6 bộ lọc Gabor định hướng** → mã CompCode là hướng đáp ứng trội tại mỗi pixel → **histogram 6 bin theo từng block** → nối thành feature → chuẩn hóa L2 → classifier. **Competitive Code là phần mã hóa đặc trưng, không tự nó là SVM hoặc k-NN.**

Tham số tác giả báo cáo (paper §6.3): `σx=4.85`, `σy=9.31`, `f=0.071`, kích thước cạnh block `p=14`, `λ=1.35` cho CRC_RLS. Kích thước `p=14` biểu thị cạnh block tính bằng pixel, không phải lưới 14×14 block. Cần kiểm tra quy tắc xử lý biên ảnh và phép chọn mã hướng qua bài báo, mã nguồn tham chiếu trước khi đánh giá tính tương đương.

**Mã nguồn tham chiếu:** liên kết `CR_CompCode.rar` do tác giả công bố: <https://cslinzhang.github.io/ContactlessPalm/CR_CompCode.rar>. Trước khi sử dụng cần kiểm tra khả năng truy cập, tính toàn vẹn, quyền sử dụng và sự phù hợp giữa mã nguồn với mô tả thuật toán.

### 3.2. Các bước và tiêu chí kiểm chứng

1. Đọc Official ROI (theo định dạng thực tế) → grayscale 128×128, giữ dtype/pixel-range nhất quán.
2. Tạo 6 kernel Gabor theo đúng định nghĩa/pha/chiều định hướng và đáp ứng quy định trong paper; không dùng kernel OpenCV tùy ý rồi nhận tương đương mã MATLAB gốc.
3. Lấy `argmin` đáp ứng của các kernel theo paper → CompCode map các giá trị `{0,1,2,3,4,5}`.
4. Chia không gian theo **kích thước block pixel**; histogram 6 giá trị cho mỗi block → nối vector; công bố số chiều đo từ code, chính sách biên và chế độ chuẩn hóa.
5. Chuẩn hóa feature về L2-unit; cùng vector đầu ra `feature.npy`/cache (có dấu vết cấu hình và mã nguồn) dùng cho **M1, M2, M3**. Không fit preprocessing học từ dữ liệu toàn bộ test.

**F0 — tiêu chí nghiệm thu đặc trưng:** Thử nghiệm ảnh tổng hợp theo 6 hướng; kiểm tra kích thước, phạm vi mã, tính lặp lại và giá trị hữu hạn; đối chiếu biểu diễn trung gian với mã tham chiếu khi có thể; lưu cấu hình và mã định danh commit nguồn. Trường hợp khác biệt phương pháp, dùng tên **“CompCode histogram triển khai theo Zhang et al. (2017)”**, không ghi là tái hiện nguyên bản.

---

## 4. CÁC MÔ HÌNH NHẬN DẠNG

| ID | Tên chính thức | Input | Điều kiện so sánh |
|---|---|---|---|
| **M1** | **1-NN nearest-template** | L2 CompCode histogram | Baseline học không tham số từ Ch. 4 |
| **M2** | **Linear SVM đa lớp** | Cùng L2 CompCode histogram | Baseline phân biệt từ Ch. 5 |
| **M3** | **CRC_RLS** | Cùng L2 CompCode histogram | Collaborative representation, gắn với paper Tongji |
| **M4** | **ResNet18 transfer learning** | Cùng Official ROI, xử lý ảnh cho pretrained CNN | So sánh pipeline learned vs handcrafted, không ghép Gabor |

### M1 — 1-NN nearest-template

Gallery chứa 8 ảnh/class khi dev, 10 ảnh/class khi final. Với mọi lớp `c`, `d_c(y) = min_{j∈gallery(c)} ||y-v_j||₂`. Rank-1 = lớp có `d_c` nhỏ nhất; Top-5 = **5 lớp khác nhau** có `d_c` nhỏ nhất. Với vector L2-unit, Euclidean và cosine có thứ tự tương ứng.

**Lý do chọn `k=1`:** nhất quán ranking theo lớp; thử `k=3/5` chỉ như sensitivity phụ nếu định nghĩa scoring/voting và top-k nhất quán. Không dùng `kNN.predict` với vote k=5 rồi lập Top-5 bằng quy tắc mâu thuẫn.

**Artifact:** gallery vectors, labels, extractor config, kết quả CV và benchmark thời gian tìm kiếm.

### M2 — Linear SVM đa lớp

Dùng `LinearSVC` (One-vs-Rest theo cấu hình được xác định), `C∈{0.1,1,10}`; căn cứ CV trung bình + tie-break theo độ phức tạp và thời gian suy luận theo tiêu chí xác định trước. **Ba mô hình M1–M3 sử dụng cùng đặc trưng L2-unit trong phép so sánh chính**. `StandardScaler` nếu thử phải là một **run phụ được báo rõ**, fit mỗi fold train, không làm thay đổi phép so sánh chính giữa classifier.

Ghi đúng loss (ví dụ squared-hinge nếu `LinearSVC` cấu hình vậy), solver, convergence warning, số iteration, phiên bản thư viện. Class score = `decision_function`; **margin không phải xác suất hiệu chuẩn**.

**Artifact:** model `.joblib`, mapping class IDs, feature config, bảng CV và thời gian huấn luyện/suy luận.

### M3 — CRC_RLS (Collaborative Representation via Regularized Least Squares)

Dictionary `A=[v₁,...,vₙ] ∈ ℝ^{m×n}` được lập từ feature L2-unit của gallery; truy vấn `y∈ℝ^m`.

\[
\hat x=\underset{x}{\arg\min}\;\|y-Ax\|_2^2+\lambda\|x\|_2^2,\qquad
\hat c=\underset{c}{\arg\min}\;\|y-A\delta_c(\hat x)\|_2^2.
\]

- Dùng nghiệm ridge ổn định số: `x = A.T @ solve(A @ A.T + λI, y)` khi dimension feature `m` nhỏ hơn số gallery `n`; tránh nghịch đảo trực tiếp ma trận `6000×6000` và tránh giải lại từng truy vấn. Precompute/factorize một lần sau khi xây gallery.
- **Mỗi lớp** có reconstruction residual, sắp xếp tăng theo residual để lấy Rank-1/Top-5; `λ>0`, bắt đầu từ `1.35` theo paper, thử một lưới **nhỏ cố định** trên validation.
- Kiểm chứng nghiệm tương đương công thức ridge trực tiếp trên một bài toán nhỏ; kiểm chứng residual lớp, không NaN, nhãn, batch prediction và thời gian train/precompute.
- Chỉ gọi **CR_CompCode reproduction** nếu cả feature và classifier tương đương mã/thiết lập tác giả. Nếu không, gọi **CompCode-hist + CRC_RLS (paper-inspired)**.

**Artifact:** dictionary, nhãn, phép factorization/precompute, config λ, kết quả CV/final, benchmark latency tổng.

### M4 — ResNet18 CNN (phần mở rộng học sâu)

- Pretrained ResNet18 ImageNet, thay classifier cuối bằng 600 lớp; Official ROI grayscale → 224×224, 3 kênh lặp nếu dùng pretrained RGB, chuẩn hóa đúng weights API.
- **Giao thức CNN:** Huấn luyện tầng phân loại cuối trước; đánh giá tùy chọn fine-tune `layer4` với learning rate thấp hơn trên session1-validation. M4 yêu cầu huấn luyện và suy luận bằng ResNet18; phương án **trích embedding pretrained + SVM** được phân loại là pipeline khác.
- Sử dụng một validation split cố định 8/2 trên session1 (trùng fold 0 của các mô hình cổ điển) để lựa chọn tham số và early stopping. Dự kiến chạy **3 seed** nếu tài nguyên GPU cho phép; xác định trước lịch huấn luyện cuối từ giai đoạn phát triển, huấn luyện trên 6.000 ảnh session1 và đánh giá **tất cả seed** trên session2, không lựa chọn seed theo kết quả kiểm thử.
- Augmentation chỉ trên training ROI: sáng/contrast và góc quay nhỏ, có kiểm tra không cắt mất vùng vân; không horizontal flip tùy tiện. Log training loss, validation Rank-1, epoch, model weights/version, GPU, wall time.
- Dùng Kaggle/Colab GPU nếu local không có GPU; không mặc định tài nguyên miễn phí luôn sẵn. Lưu trọng số/seed/artifact sau huấn luyện.
- **So sánh trung thực:** M4 dùng ảnh 224×224 còn classical dùng CompCode trên 128×128; đây là so sánh **toàn pipeline**, chịu ảnh hưởng của biểu diễn/kiến trúc/kích thước, không thể quy toàn bộ chênh lệch Accuracy cho riêng kiến trúc CNN.

**M0 — tiêu chí nghiệm thu mô hình:** Huấn luyện và suy luận trên dữ liệu hợp lệ; lưu và nạp artifact cho kết quả nhất quán; ánh xạ nhãn 1–600; xuất điểm và thứ hạng cho **600 lớp**, với nhãn dự đoán trùng Top-1.

---

## 5. THIẾT KẾ VÀ MA TRẬN THỰC NGHIỆM

| ID | Thiết kế | Biến được kiểm soát / câu hỏi | Output cần có | Ưu tiên |
|---|---|---|---|---|
| **E0** | Dataset audit + EDA + split | Dữ liệu đủ, ROI đúng, có sai lệch chất lượng giữa session? | Manifest, 5–7 hình/bảng EDA, fold CSV | **P0** |
| **E1** | Extractor CompCode, unit tests/visualization | Đúng hướng, đúng block, khớp paper/mã nguồn đến mức nào? | Feature-map/histogram, shape, test, ghi sai khác | **P0** |
| **E2** | **M1 vs M2 vs M3** trên cùng feature | Thuật toán quyết định nào tốt nhất khi feature cố định? | CV results; bảng final; CMC/confusion | **P0** |
| **E3a** | Global histogram vs block-wise histogram, **giữ classifier CRC_RLS** | Lợi ích của biểu diễn có bố cục không gian? | Ablation bảng và phân tích khác chiều feature | **P1** |
| **E3b** | HOG vs CompCode, **giữ SVM** | Phân tích thêm lý do chọn CompCode; không bắt HOG thành mô hình chính | Kết quả CV + một bảng đối chứng nhỏ | **P1** |
| **E4** | **ResNet18** (seed / head vs fine-tune) | CNN học đặc trưng có lợi gì so với classical? | Curves, 3 seed nếu khả thi, final metrics | **P1 — M4** |
| **E5** | Error analysis + quality + latency | Khi nào nhận sai? Độ chính xác/độ trễ đánh đổi ra sao? | Ca đúng/sai, nhóm lỗi, timing CPU, model size | **P0** |
| **E6** | Robustness synthetic (rotation/brightness/blur) | Kiểm tra độ bền với các mức nhiễu tổng hợp được quy định trước | Đường suy giảm hiệu năng, không điều chỉnh tham số theo test | P2 |

**Ngoài phạm vi chính:** HOG–Gabor fusion, các classifier bổ sung, PalmNet/CCNet, ROI tự trích xuất, webcam và verification EER. Các nội dung này chỉ xem xét sau E0–E5. Thực nghiệm E3b đánh giá lựa chọn đặc trưng trên session1; kết quả cuối trên session2 không phục vụ chọn lại biểu diễn.

### Quy trình lựa chọn mô hình và ngăn ngừa rò rỉ dữ liệu

1. **Pre-register**: cố định danh sách thí nghiệm E0–E6, seeds/folds, tham số ứng viên, metric chính và logic tie-break trước khi chạy session2.
2. Chạy E1 + E2 trên 5-fold session1. Ví dụ grid: M1 `k=1`/Euclidean cố định; M2 `C={0.1,1,10}`; M3 `λ={0.1,1.35,10}`; feature block ban đầu theo paper, chỉ khảo sát kích thước block **trong CV** nếu có lý do.
3. Lựa chọn config theo **mean CV Rank-1**; nếu gần bằng nhau chọn ít phức tạp/nhanh hơn. **Điểm CV không phải ước lượng độc lập của kết quả cuối** vì đã dùng nó chọn cấu hình.
4. Đối chứng E3a/b thực hiện trong session1; chọn **feature chính được chọn trên validation** trước test. Nếu CompCode không vượt HOG thì vẫn có thể giữ CompCode là phương pháp nghiên cứu chính và báo cáo hạn chế; **không bắt buộc thao tác thay đổi kế hoạch theo kết quả session2**.
5. Huấn luyện/lập gallery cuối trên session1; lưu `experiment_freeze.yaml`, commit SHA, config hash, model artifact hash.
6. Thực hiện một lần đánh giá cuối theo cấu hình cố định trên session2 và tạo file prediction cho 600 lớp; chỉ phân tích kết quả, không chỉnh threshold/hyperparameter rồi gọi lần đo sau là độc lập.
7. Đối với E6, định nghĩa biến đổi/mức độ **trước final test**; chỉ thực hiện sau main result như stress test định sẵn, không dùng để sửa mô hình chính.

### Cấu trúc bảng báo cáo kết quả

| Mô hình | CV Rank-1 | Test Rank-1 | Test Rank-5 | Macro-F1 | Latency median / P95 (ms) | Artifact MB |
|---|---:|---:|---:|---:|---:|---:|
| 1-NN | TBD | TBD | TBD | TBD | TBD | TBD |
| Linear SVM | TBD | TBD | TBD | TBD | TBD | TBD |
| CRC_RLS | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet18 | TBD | TBD | TBD | TBD | TBD | TBD |

Paper Tongji công bố `CR_CompCode Rank-1 = 98.78%`, `12.48 ms/identification` trên thiết bị/phần mềm của họ. **Đặt ở bảng literature riêng** kèm điều kiện, không trộn với kết quả nhóm, không dùng để cam kết nhóm sẽ đạt mức ấy.

---

## 6. METRICS, THỐNG KÊ VÀ PHÂN TÍCH LỖI

### Chỉ số và cách tính

- **Rank-1** = số probe dự đoán đúng Palm ID / số probe (trong closed-set 600 lớp bằng accuracy top-1).
- **Rank-5** = tỷ lệ nhãn đúng nằm trong **Top-5 Palm ID phân biệt**, không phải Top-5 ảnh gallery. Mọi mô hình phải trả classwise score cho đủ 600 lớp. k-NN/CRC residual sort tăng; SVM/logits sort giảm; tie-break theo Palm ID được cố định.
- **Macro-Precision/Recall/F1** tính trên toàn 600 lớp, dùng `zero_division=0`; confusion matrix đầy đủ 600×600 và heatmap nhóm lỗi nổi bật.
- **CMC** từ classwise ranks: Rank-1, 2, 5, 10, 20 (hoặc toàn đường nếu tiện).
- **Train/feature/inference time**: phân tách preprocess, feature (nếu dùng), model matching, total; đo warm-up, batch, median/P95 và môi trường. Muốn so tốc độ công bằng thì đo tất cả trên **cùng CPU** (CNN CPU inference benchmark riêng); GPU CNN báo cáo bảng phụ, không đặt chung cột tốc độ CPU.
- **CI cho Rank-1 và chênh lệch paired**: bootstrap **theo Palm ID** (600 cluster), resample Palm IDs và giữ 10 probe trong cluster; báo mức 95%, số lần bootstrap/seed. Hai tay cùng người có thể còn tương quan nhưng thiếu ánh xạ Person ID được xác minh để cluster theo người. Không giả định 6.000 ảnh độc lập hoàn toàn.
- **CNN**: báo cáo từng seed và mean±std (nếu chạy nhiều seed); không lựa chọn seed dựa trên kết quả test.

### Phân tích sai số và trường hợp nhận dạng sai

Lấy ít nhất **10–15 case sai**, **5 case đúng** rải từ nhiều Palm ID: ảnh probe, ảnh đối chiếu gallery, nhãn thật/dự đoán, top-5, score thô và quality proxies. Phân nhóm có dẫn chứng: ROI lệch, đường vân mờ, khác độ sáng, mẫu các lòng bàn tay có cấu trúc tương tự. **Không kết luận nhân quả từ mỗi hình minh họa**; giả thuyết nguyên nhân chỉ được gọi là xác nhận khi có đối chứng.

**S1 — tiêu chí nghiệm thu đánh giá:** Với mọi probe, `pred_palm_id` bằng phần tử đầu tiên của `top5_palm_ids` trong `predictions.csv`; kiểm thử công thức metric bằng dữ liệu giả lập; các mô hình sử dụng cùng tập `sample_uid`; bootstrap có thể tái lập; bảng kết quả được tạo từ `predictions.csv`.

---

## 7. DEMO, TỔ CHỨC MÃ NGUỒN VÀ KHẢ NĂNG TÁI LẬP

### Tổ chức repository

Cấu trúc tệp tương ứng với mã nguồn và lệnh thực thi:

```text
palmprint-recognition/
├── README.md
├── pyproject.toml
├── configs/
│   ├── dataset.yaml
│   ├── compcode.yaml
│   ├── models.yaml
│   └── resnet18.yaml
├── data/
│   ├── README.md
│   └── raw/                      # session1/, session2/ (không commit dữ liệu)
├── src/palmprint/
│   ├── cli.py
│   ├── cli_cnn.py
│   ├── utils.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preparation.py
│   ├── features/
│   │   ├── compcode.py
│   │   └── extraction.py
│   ├── models/
│   │   ├── classical.py
│   │   └── resnet.py
│   ├── training/
│   │   ├── selection.py
│   │   ├── classical.py
│   │   └── cnn.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── analysis.py
│   │   └── cnn.py
│   └── reporting/
│       ├── evidence.py
│       └── summary.py
├── scripts/run_smoke.py        # synthetic integration test
├── tests/                      # unit and integration tests
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results_analysis.ipynb
├── app/main.py                 # Streamlit
├── experiments/registry.yaml
├── artifacts/                  # model và khóa thí nghiệm (không commit)
├── outputs/                    # CV, cache, kết quả từng run (không commit)
├── reports/                    # hình, bảng, khung báo cáo và slide
└── docs/                       # phương pháp, protocol, training, kết quả, bảo mật
```

Tên và cấu trúc thực tế của các tệp báo cáo đầu ra: [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md).

**Hồ sơ thực nghiệm theo mã nguồn hiện tại:**

| Nhóm dữ liệu | Đường dẫn | Nội dung |
|---|---|---|
| Validation (M1–M3) | `outputs/cv/<representation>_<model>.csv`, `*_selection.json`, `*_oof_predictions.csv` | Fold, tham số, kết quả OOF và cấu hình được chọn |
| Artifact (M1–M3) | `artifacts/<model>.joblib`, `<model>_freeze.json`, `protocol_lock.json` | Model, hash và trạng thái khóa ba mô hình cổ điển |
| Đánh giá cuối | `outputs/runs/<model>/metrics.json`, `predictions.csv`, `all_class_scores.npz` | Chỉ số, từng dự đoán và điểm toàn bộ lớp |
| Phân tích lỗi | `per_class_metrics.csv`, `cmc_full.csv`, `confusion_matrix.npy`, `errors.csv`, `class_order.json` trong thư mục run | F1/lỗi theo lớp, CMC và thứ tự lớp |
| Huấn luyện CNN | `outputs/runs/resnet18_seedN/` và `artifacts/resnet18_seedN.pth` | `training_config.json`, `dev_history.csv`, `final_history.csv`, checkpoints |
| Bảng và biểu đồ | `reports/tables/`, `reports/figures/` | Kết quả tổng hợp và biểu đồ được sinh từ run evidence |

Cấu hình, hash và môi trường được ghi trong metadata tương ứng; mã nguồn **không tạo `config.yaml` hoặc `train_log` cho mọi classical run**. `predictions.csv` không chứa latency từng ảnh; `metrics.json` lưu thời gian suy luận theo batch, còn `benchmark_cpu.json` ghi phép đo một ảnh. Git commit SHA không tự động được lưu khi chưa có trường tương ứng trong mã nguồn. Xem [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md) để biết đầy đủ hợp đồng output.

**Unit test tối thiểu:** `00001→Palm1`, `00010→Palm1`, `00011→Palm2`, `06000→Palm600`; không trùng sample UID; test ROI decode; CompCode 6 codes; CRC residual toy; top-5 5 lớp duy nhất; serialized artifact và source predictor cho cùng kết quả; không dùng true label/filename để suy ra prediction trong app.

### Kịch bản demo khoảng 90 giây

1. Chọn ảnh ROI từ session2 hoặc upload ảnh **có định dạng ROI phù hợp**; không hỗ trợ nhận dạng trực tiếp từ ảnh bàn tay gốc ngoài quy trình ROI đã quy định.
2. Chọn một trong M1–M4; hiển thị **Palm ID**, Top-5 lớp, ảnh gallery của các lớp tương ứng, score khoảng cách/margin/residual (không gọi là % xác suất nếu chưa hiệu chuẩn), thời gian.
3. **Phạm vi giao diện hiện có:** chọn **một mô hình mỗi lượt** trong số artifact khả dụng; người dùng có thể giữ nguyên ảnh truy vấn và đổi lựa chọn để quan sát kết quả theo từng mô hình. Chưa có nút so sánh đồng thời bốn mô hình hoặc panel chất lượng ảnh trong ứng dụng.
4. **Phân tích kết quả nằm ngoài giao diện dự đoán hiện tại:** `palmprint report` đọc `outputs/runs/` và sinh bảng trong `reports/tables/`, hình trong `reports/figures/`. Ứng dụng Streamlit chưa tích hợp dashboard CMC/confusion hoặc báo cáo tương tác; dùng các tệp xuất để thuyết trình.
5. Chuẩn bị tối thiểu 3 ca kiểm tra trước buổi bảo vệ: ảnh nhận đúng, ảnh nhận sai có giải thích và ảnh đầu vào hỏng; tư liệu minh họa dự phòng chỉ bổ trợ cho demo chạy thực tế.

**S2 — tiêu chí nghiệm thu demo:** Chạy trên thiết bị khác theo README; sử dụng artifact mô hình và suy luận thực tế; thời gian phản hồi phù hợp trình bày và có cơ chế xử lý tệp không hợp lệ.

---

## 8. BÁO CÁO, SLIDE VÀ CHỨNG MINH ĐÓNG GÓP

### Dự kiến báo cáo kỹ thuật 26 trang (theo yêu cầu 20–30 trang)

| Chương | Số trang dự kiến | Trọng tâm |
|---|---:|---|
| 1. Bài toán và tổng quan | 3 | 1:N, 600 lớp, câu hỏi RQ1–RQ4, paper gốc |
| 2. Cơ sở lý thuyết | 5 | CompCode/Gabor, k-NN, SVM, CRC_RLS, CNN; công thức **đúng code** |
| 3. Dataset và tiền xử lý | 4 | Tongji, EDA, official ROI, folds, leakage, audit |
| 4. Phương pháp và thực nghiệm | 6 | Bốn mô hình, CV grid, giao thức, chính sách cố định giao thức kiểm thử |
| 5. Kết quả và phân tích | 6 | Table final, CMC/confusion, ablation, error, CI, latency |
| 6. Kết luận, giới hạn | 2 | Trả lời từng RQ; học được gì; giới hạn cỡ mẫu/thiết bị/pilot |
| **Tổng** | **26** | Trang bìa/mục lục/tài liệu tham khảo tính theo yêu cầu thực tế |

**Cấu trúc slide dự kiến (13 trang):** (1) đề tài; (2) bài toán; (3) Tongji và 600 lớp; (4) EDA; (5) protocol; (6) CompCode; (7) k-NN/SVM/CRC_RLS; (8) ResNet18; (9) bảng kết quả và CI/latency; (10) confusion matrix và CMC; (11) ablation; (12) ca sai và demo; (13) kết luận/giới hạn. Xem `reports/slides/SLIDE_OUTLINE.md`.

**Nội dung đánh giá điểm thưởng sáng tạo:** Thể hiện khả năng triển khai có thể kiểm chứng, so sánh thuật toán trên cùng đặc trưng, kiểm định ảnh hưởng của histogram theo block, đối chứng CNN và phân tích độ chính xác – độ trễ. Mức điểm do giảng viên đánh giá theo rubric.

---

## 9. TIẾN ĐỘ DỰ KIẾN VÀ MỐC NGHIỆM THU

*18 ngày là khối lượng ước lượng để nhóm tự lập lịch; không thay thế deadline thực tế của môn.*

| Giai đoạn | Thời gian gợi ý | Việc cần bàn giao | Gate |
|---|---|---|---|
| D0–D1 | Ngày 1–3 | GV duyệt nguồn, manifest, EDA, folds và unit tests | **G0: Dataset** |
| F0 | Ngày 4–6 | CompCode theo paper, test feature-map/block/histogram, lưu cache | **G1: Features** |
| M1–M3 + E3 | Ngày 7–10 | 5-fold CV, E3 ablation chỉ trên session1, chọn params, unit test CRC, đủ ba classifier | **G2: Classical models** |
| M4 | Ngày 8–12 song song | CNN train+validation, seeds, checkpoints, curves | **G3: CNN** |
| Freeze & final | Ngày 13 | Cố định config/code/splits sau E3a/E3b, huấn luyện lại và đánh giá trên session2 | **G4: Final benchmark** |
| Analysis & app | Ngày 14–16 | Tổng hợp E3 trên validation; thực hiện E5, CI, figures, demo load artifact | **G5: Evidence and demo** |
| Nộp & bảo vệ | Ngày 17–18 | Report 20–30 trang, slide, README, phân công, GitHub/Classroom | **G6: Submission** |

**Phương án điều chỉnh phạm vi:** Ưu tiên D0–D1, M1–M4, metrics, demo và báo cáo. Có thể giảm quy mô E3b và E6; kiểm thử leakage và quy trình lựa chọn mô hình tách biệt với test vẫn giữ nguyên. G0 chỉ đạt khi có xác nhận nguồn dữ liệu của giảng viên.

### Phân công công việc dự kiến cho nhóm 3 người

- **Thành viên A — Data/Feature:** duyệt nguồn, manifest/EDA, split, Official ROI, mã hóa CompCode, kiểm chứng feature và kiểm thử.
- **Thành viên B — Classical Modeling/Evaluation:** k-NN, SVM, CRC_RLS, CV, chọn tham số, thống kê, confusion, error analysis.
- **Thành viên C — CNN/Product:** ResNet18, checkpoint, demo giao diện, liên kết model, benchmark CNN, slide/báo cáo.
- **Cả nhóm:** thiết kế protocol, code review, đối chiếu paper, viết kết luận, diễn tập phản biện, GitHub commits/bảng phân công.

Nếu chỉ có 2 người: A phụ trách data + features + M1/M2; B phụ trách M3/M4 + evaluation + app; phần report và QA cùng chịu trách nhiệm. Phân công trong báo cáo phản ánh công việc thực tế.

---

## 10. RỦI RO TRIỂN KHAI VÀ CÂU HỎI PHẢN BIỆN

| Rủi ro | Biện pháp bắt buộc |
|---|---|
| Feature khác bản chính thức | Dùng paper + link mã tác giả để kiểm chứng; ghi rõ **paper-inspired** nếu không tái hiện đủ |
| Khác biệt session | CV nội bộ trên session1; không suy ra test từ CV; đánh giá cross-session chỉ bằng session2 |
| Giới hạn tính độc lập của session2 | Ghi rõ thứ tự classical, CNN logits và embedding; không tiếp tục tune bằng test |
| `CRC_RLS` cấp phát ma trận lớn | Dùng ridge dual/Cholesky trong không gian chiều feature, unit test tính đúng và memory |
| CNN overfit, thiếu GPU | Pretrained transfer learning, early stopping trên session1, 3 seed nếu có tài nguyên; dùng Kaggle/Colab |
| Chênh lệch độ phân giải CNN/classic | So pipeline, không quy chênh lệch trực tiếp cho classifier hoặc CNN alone |
| Kết quả chưa có phân tích định lượng | Lưu predictions, lỗi, CMC, ablation, bootstrap và truy vết số liệu từ code |
| Demo thiếu độ ổn định | Dùng cùng artifact với thực nghiệm, giới hạn input theo định dạng ROI, thử nghiệm ca đúng/sai và tệp không hợp lệ |
| Nguy cơ lộ dữ liệu sinh trắc | Dataset theo nguồn/điều khoản; không đẩy toàn bộ ảnh private vào repo public |

**Câu hỏi phản biện dự kiến:**

1. *Tại sao không dùng pixel trực tiếp?* — CompCode tập trung vào cấu trúc định hướng; phải chứng minh bằng E3b/đối chứng hoặc giải thích hạn chế nếu không vượt.
2. *Tại sao 3 mô hình cùng một feature?* — Để phân tích tác động thuật toán quyết định trong RQ1; M4 là so sánh hệ thống learned features.
3. *CRC_RLS khác k-NN ở điểm nào?* — k-NN dựa vào mẫu gần nhất, CRC dùng biểu diễn kết hợp toàn gallery và residual theo lớp.
4. *Vì sao CNN không dùng CompCode?* — CNN học đặc trưng end-to-end từ ảnh; ép thủ công biến đổi sẽ là thí nghiệm hybrid khác, không cần thiết.
5. *Điểm số paper là của ai?* — 98,78% Rank-1 là của Zhang et al. 2017; nhóm chỉ báo cáo số đo từ run có thể tái lập.
6. *Vì sao session1 và session2 có cùng lớp?* — Đây là closed-set identification, yêu cầu nhận đúng một Palm ID có trong gallery; khác bài toán unseen-subject generalization.
7. *Validation có leakage không?* — Cùng lớp train/val là hợp lệ; nhưng không có trùng ảnh, feature fitting chỉ trên train-fold, toàn bộ session2 ngoài vòng chọn tham số.
8. *Đồ án có giới hạn gì?* — 300 người với điều kiện chụp tương đối kiểm soát, chỉ 600 palms, chưa chứng minh khả năng nhận dạng người lạ hoặc ảnh điện thoại/webcam, các so sánh bổ sung diễn ra sau khi đã quan sát test.

---

## 11. DANH MỤC KIỂM TRA TRƯỚC KHI NỘP

- [ ] GV xác nhận nguồn dataset; ghi nhận minh chứng hợp lệ.
- [ ] Manifest đủ nhãn/ảnh/quality và audit duplicates; `session/stem` là sample UID.
- [ ] 5-fold session1 với 8/2 ảnh/Palm ID, cùng folds cho classical, có kiểm thử leakage.
- [ ] CompCode 6 hướng + block histograms đúng logic, ghi xử lý biên, shape, config và kết quả test.
- [ ] Hoàn thành M1 (1-NN), M2 (SVM), M3 (CRC_RLS) với cùng bộ đặc trưng CompCode histogram.
- [ ] Huấn luyện và kiểm thử suy luận M4 (ResNet18); lưu loss/accuracy, checkpoints, seed và cấu hình phần cứng.
- [ ] Chọn tham số chỉ trên session1 và lưu cấu hình cố định trước khi đánh giá session2.
- [ ] Main final test session2 có 600 class scores/probe, Rank-1, Rank-5, Macro-F1, CMC/confusion.
- [ ] Thực hiện E3a (block ablation), ưu tiên E3b (đối chứng đặc trưng), tổng hợp tối thiểu một bảng ablation.
- [ ] Paired cluster bootstrap; bảng CPU latency, artifacts và 10–15 phân tích ca sai.
- [ ] Nêu lịch sử quan sát session2 và phạm vi hậu nghiệm của các so sánh bổ sung.
- [ ] Demo chọn một artifact đã huấn luyện, hiện Top-5 Palm IDs/ảnh gallery cho mẫu thuộc manifest và có 3 tình huống kiểm thử; upload tự do không có gallery đối chiếu trong giao diện hiện tại.
- [ ] Báo cáo 20–30 trang, tài liệu tham khảo, 13 slide, bảng phân công, code có README, link GitHub/Classroom.
- [ ] Số liệu trong report/slide lấy từ `outputs/runs/` và các bảng/hình trong `reports/`; không sử dụng số paper như kết quả của nhóm.

**Tiêu chí hoàn tất:** Các hạng mục tương ứng rubric có minh chứng, kết quả thực nghiệm truy xuất được từ mã nguồn và artifact, demo hoạt động trên dữ liệu hợp lệ, báo cáo/slide nhất quán với số liệu. Số mô hình và độ chính xác đơn lẻ không thay thế yêu cầu đánh giá tổng thể.

## 12. TÀI LIỆU THAM KHẢO

1. **Đề giảng viên**: `BÀI TẬP LỚN CUỐI KỲ.pdf` (file nhóm cung cấp) — rubric, sản phẩm cần nộp, nhóm và đề tài.
2. **Giáo trình**: Duda, Hart, Stork. *Pattern Classification*, 2nd ed., Wiley — Ch. 4, 5, 6, 9; đối chiếu mục lục từ PDF nhóm cung cấp.
3. **Paper chính**: Zhang, L., Li, L., Yang, A., Shen, Y., Yang, M. (2017). *Towards contactless palmprint recognition: A novel device, a new benchmark, and a collaborative representation based identification approach*. **Pattern Recognition 69**, 199–212. DOI: [10.1016/j.patcog.2017.04.016](https://doi.org/10.1016/j.patcog.2017.04.016). [Toàn văn](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).
4. **Nguồn dataset và mã tham chiếu**: <https://cslinzhang.github.io/ContactlessPalm/>; [CR_CompCode archive](https://cslinzhang.github.io/ContactlessPalm/CR_CompCode.rar) — cần kiểm tra khả năng tải và các điều khoản sử dụng khi làm.
5. **Nền tảng Competitive Code**: Kong, W. K., Zhang, D. (2004). *Competitive coding scheme for palmprint verification*, 17th International Conference on Pattern Recognition — chỉ trích các nhận định đúng với bài gốc.
6. **CNN tham khảo**: He, K. et al. (2016). *Deep Residual Learning for Image Recognition*, CVPR. [arXiv](https://arxiv.org/abs/1512.03385).

**Nguyên tắc trích dẫn:** Phân biệt công thức, thông số và kết quả từ tài liệu tham khảo với các cấu hình và số đo của nhóm; nêu rõ nguồn và điều kiện thí nghiệm tương ứng.
