# Giao thức nghiên cứu và tiêu chí nghiệm thu

### Tiền xử lý Official ROI

- Dùng **Official ROI 128×128** của Tongji làm đầu vào. Ảnh Original chỉ phục vụ EDA, không được đưa trực tiếp vào classifier.
- Một hàm `read_gray` dùng chung: giải mã bằng OpenCV, chuyển ảnh 8-bit về grayscale; kiểm tra hình dạng 128×128. Không tự thêm CLAHE, blur, denoise, threshold hay cắt lại ROI trong phép so sánh chính.
- Nhánh truyền thống: grayscale 128×128 → CompCode histogram → classifier. Nhánh CNN: cùng grayscale đó → resize 224×224 → tensor/chuẩn hóa theo ImageNet; augmentation nhẹ chỉ ở train.
- Dùng một quy trình đọc ảnh cho cả train, validation, test và demo; các biến thể xử lý ảnh chỉ xuất hiện trong ablation được định nghĩa trước.


## 1. Đối tượng nhận dạng

- Bài toán **closed-set identification**, nhãn là **Palm ID** (1…600), không phải Person ID.
- Data chính: **Tongji Official ROI**, không sử dụng ảnh gốc làm đầu vào mô hình.
- Mỗi Palm ID được dự kiến có 10 ảnh trong session1 và 10 ảnh trong session2.
- Kết quả paper gốc là số từ nguồn tham khảo, không thay thế thực nghiệm do nhóm thực hiện.

## 2. Sơ đồ dữ liệu

| Công đoạn | Session | Số mẫu nếu đầy đủ | Cách dùng |
|---|---|---:|---|
| CV-fold train | session1 | 4.800 | fit mọi bước có học, model và gallery |
| CV-fold val | session1 | 1.200 | so tham số/hyperparameters |
| Final train | session1 | 6.000 | fit/galleries với tham số đã chọn |
| Final test | session2 | 6.000 | chỉ đánh giá, không chọn mô hình |

Folds: `seed=42`, từng Palm ID được shuffle một lần, chia 5 phần bằng nhau. Mỗi ảnh session1 xuất hiện trong validation đúng một fold. Đối với ResNet18 dùng fold0 (8/2) để chọn epoch, sau đó reinitialize và huấn luyện toàn session1 theo schedule cố định. Trong pipeline cổ điển, code không đọc ảnh session2 ở các bước CV/final train; chỉ sau khi cả ba model được khóa chung mới cho phép trích test features. Pilot lịch sử vẫn làm suy yếu tuyên bố test hoàn toàn độc lập.

**Lịch sử test:** Chưa có bằng chứng xác nhận pilot HOG được nhắc trong bản tài liệu ban đầu. Benchmark classical và CNN đã quan sát session2 trước khi bổ sung embedding; các so sánh bổ sung là hậu nghiệm. Không dùng test để chọn cấu hình hoặc seed.

## 3. Registry và tiêu chí lựa chọn

- M1: 1-NN theo squared Euclidean trên normalized CompCode (chỉ k=1).
- M2: LinearSVC C∈{0.1,1,10}, squared-hinge, OvR, dual=False, không thêm StandardScaler trong phép so sánh chính.
- M3: CRC_RLS λ∈{0.1,1.35,10}, ridge + class reconstruction residual.
- Chọn tham số bằng **mean 5-fold validation Rank-1**. Tie-break: tham số nhỏ hơn. CV std mô tả biến thiên fold, không phải CI độc lập.
- M4: ResNet18 pretrained, validation fold0, head trước, fine-tune layer4 nếu được thử, 3 seed 17/29/43; không chọn seed bằng test.

## 4. Ma trận thí nghiệm

| Mã | Thực nghiệm | Biến kiểm soát | Kết quả |
|---|---|---|---|
| E0 | Audit, EDA, splits | Tongji Official ROI | manifest, histograms, quality |
| E1 | CompCode | kernel/orientation map, block pixels=14 | 6 mã, L2 norm, dim tự đo |
| E2 | k-NN/SVM/CRC_RLS | giống feature và folds | CV + cross-session test |
| E3a | Global vs block-wise histogram | giữ CRC_RLS | session1 CV ablation |
| E3b | HOG vs CompCode | giữ LinearSVM | session1 CV ablation |
| E4 | ResNet18 head/fine-tune | cùng ROI, fold0 | training histories, seed results |
| E5 | Error/quality/latency | cùng probe | examples, proxies, med/P95 |
| E6 | Synthetic robustness | model frozen; xoay5°, ±15 brightness, blur3 | stress performance, secondary |

E3a/E3b không dùng session2 để quyết định chọn feature; E6 chỉ diễn giải sau main result, không retune.

## 5. Chỉ số

- `Rank-1`: predicted Palm ID đúng.
- `Rank-5`: Palm ID thật trong 5 **lớp riêng biệt** xếp đầu. Giữ ranking nhất quán trên 600 class scores.
- Macro precision/recall/F1: tính cả các lớp, `zero_division=0`.
- CMC: Rank-1/2/5/10/20 trên classwise rankings.
- Full 600×600 confusion matrix; hình top confusions được trích riêng.
- `95% CI`: bootstrap theo Palm ID, giữ tất cả probe thuộc palm trong cùng cluster; ghi rõ không có Person ID để cluster hai tay cùng người.
- CPU latency: preprocess, feature, classify và total median/P95 ở chế độ single-image; batch amortized time được báo riêng. CNN GPU/CPU báo riêng.

## 6. Điều kiện công bố kết quả

1. `prepare` strict thành công, không trùng UID hay cross-session SHA256.
2. CompCode vượt qua test shape/sanity. **Phải đối chiếu bản MATLAB gốc trước khi tuyên bố tái hiện CR_CompCode nguyên bản.**
3. Output thuộc mô hình đã khóa; đánh giá từ frozen artifact; không xóa run test để tối ưu lại.
4. Bảng số liệu sinh từ CSV/JSON, không sao chép số thí nghiệm từ paper vào bảng kết quả nhóm.
5. Nhận xét về blur/ánh sáng chỉ là tương quan chất lượng, không kết luận quan hệ nhân quả nếu thiếu đối chứng.

## 7. Tiêu chí chấm theo rubric

- 1,5đ: xác định bài toán/dataset; xác nhận GV duyệt nguồn.
- 1,5đ: tiền xử lý/đặc trưng/split.
- 3,5đ: ít nhất 2 thuật toán, trong đó có thuật toán nền tảng giáo trình; toán học và code.
- 2,0đ: so sánh có bảng, biểu đồ, confusion matrix, error analysis.
- 0,5đ: chương trình thật nhận mẫu và trả kết quả.
- +0,5đ: nghiên cứu nâng cao có chứng cứ; **CNN không tự động được điểm**.
- +0,5đ: demo/trình bày trực quan.

Điểm do giảng viên quyết định, không cam kết đạt 10/10 khi chưa có kết quả.
