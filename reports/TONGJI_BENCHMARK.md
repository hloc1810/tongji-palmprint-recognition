# Benchmark Tongji: mô hình cổ điển, CNN và embedding matching

Closed-set 1:N, 600 Palm ID. Mỗi run dùng 6.000 ảnh session1 để train/gallery và 6.000 probe session2. Các hàng CNN tổng hợp ba seed 17, 29, 43. Kết quả embedding được đọc từ metrics và đã đối chiếu score, prediction, manifest, hash checkpoint và gallery của cả sáu run.

## Bảng tổng hợp các pipeline

| Pipeline | Rank-1 | Rank-5 | Macro-F1 |
|---|---:|---:|---:|
| M1 — 1-NN nearest-template + CompCode | 94.4333% | 96.9667% | 93.5583% |
| M2 — Linear SVM OvR + CompCode | 97.3667% | 98.5167% | 96.8154% |
| M3 — CRC_RLS / Collaborative Ridge + CompCode | 98.2667% | 99.0833% | 97.9926% |
| M4-E1 — ResNet18 600-class logits | 64.3611% ± 1.0409 | 81.9111% ± 0.6545 | 60.7933% ± 1.1012 |
| M4-E2-nearest — ResNet18 512-D Nearest Cosine | 73.7444% ± 0.6241 | 87.5000% ± 0.4368 | 70.5094% ± 0.8075 |
| M4-E2-centroid — ResNet18 512-D Centroid Cosine | 68.8278% ± 0.7558 | 83.7000% ± 0.6252 | 65.3683% ± 0.8205 |

Các giá trị sau dấu ± là sample std giữa ba seed, ddof=1, theo điểm phần trăm. Đây là trung bình chỉ số của các model riêng biệt, không phải ensemble hoặc 18.000 probe độc lập. Các hàng cổ điển là một cấu hình final mỗi mô hình. CSV dùng tỷ lệ 0–1; cột delta có đơn vị điểm phần trăm.

Nearest Cosine tăng mean Rank-1 9.3833 điểm phần trăm so với logits. Centroid Cosine tăng 4.4667 điểm phần trăm. Đây là chênh lệch mô tả, chưa phải kết luận có ý nghĩa thống kê từ kiểm định ghép cặp mới.

## Embedding matching: kết quả từng seed

ResNet18 → feature 512-D → L2 normalization → gallery cosine. Dùng lại checkpoint của M4-E1 cùng seed, không train thêm. Nearest dùng cosine lớn nhất trong 10 template của mỗi Palm ID. Centroid lấy trung bình embedding đã chuẩn hóa của lớp rồi chuẩn hóa lại, tạo 600 centroid từ 6.000 ảnh gallery.

| Seed | Kỹ thuật | Rank-1 | Rank-5 | Macro-F1 | Đúng / 6.000 | CI 95% Rank-1 |
|---|---|---:|---:|---:|---:|---|
| 17 | Nearest Cosine | 73.1833% | 87.3833% | 69.7216% | 4391 | [70.3996%; 75.7350%] |
| 17 | Centroid Cosine | 68.6167% | 83.4167% | 65.1542% | 4117 | [65.6646%; 71.3167%] |
| 29 | Nearest Cosine | 73.6333% | 87.1333% | 70.4714% | 4418 | [70.9162%; 76.2504%] |
| 29 | Centroid Cosine | 68.2000% | 83.2667% | 64.6761% | 4092 | [65.2163%; 70.9338%] |
| 43 | Nearest Cosine | 74.4167% | 87.9833% | 71.3352% | 4465 | [71.5983%; 77.0512%] |
| 43 | Centroid Cosine | 69.6667% | 84.4167% | 66.2746% | 4180 | [66.7492%; 72.3500%] |

CI từng run là bootstrap theo Palm ID theo cấu hình evaluator. Không lấy trung bình hai cận CI để tạo CI cho mean ba seed. Hai kỹ thuật embedding được bổ sung sau khi đã xem kết quả session2, nên được ghi là **so sánh hậu nghiệm/exploratory**. Không dùng session2 để chọn seed hoặc tuyên bố chiến lược đã được chọn bằng validation. Chuỗi warning trong raw metrics nhắc “historical pilots” là nội dung kế thừa; bảng này chỉ khẳng định các kết quả session2 trước đó đã được quan sát.

## Thời gian embedding trên CPU theo batch

| Seed | Kỹ thuật | Trích feature probe (ms/ảnh, batch-amortized) | Cosine matching (ms/ảnh, batch-amortized) |
|---|---|---:|---:|
| 17 | nearest | 19.3375 | 0.0582 |
| 17 | centroid | 19.3375 | 0.0148 |
| 29 | nearest | 19.1878 | 0.0568 |
| 29 | centroid | 19.1878 | 0.0135 |
| 43 | nearest | 19.4693 | 0.0563 |
| 43 | centroid | 19.4693 | 0.0134 |

Feature probe được trích một lần mỗi seed và dùng chung cho hai chiến lược. Các giá trị trên là thời gian cả batch chia số ảnh, không phải median/P95 từng ảnh và không so trực tiếp với bảng độ trễ CPU cũ. Gallery đã được tạo sẵn; gallery_wall_ms trong raw metrics của lượt evaluate là thời gian nạp/kiểm tra cache, không phải chi phí tạo gallery lần đầu. Chưa có phép đo độ trễ từng ảnh cho embedding, các cột tương ứng được để trống.

## File kết quả

- [Từng run: 3 classical + 3 logits + 6 embedding](tables/tongji_benchmark.csv).
- [Mean và sample std theo pipeline](tables/tongji_benchmark_summary.csv).
- [Bảng so sánh từng run](tables/model_comparison.csv).
- [Thiết kế và kết quả CNN ban đầu](CNN_EXPERIMENTS.md).
- [Thực nghiệm bóc tách đặc trưng E3: Raw vs PCA vs CompCode](../outputs/feature_ablation/findings.md).

Nguồn của mỗi hàng mới nằm ở cột metrics_source, kèm seed, strategy, hash artifact/gallery và phạm vi timing. Kiểm chứng sáu run: outputs/embedding_benchmark/verification.json. Các file raw kết quả và model được giữ nguyên.


## Phụ lục benchmark cổ điển

5-fold session1, seed=42, 4.800 train/1.200 val mỗi fold. SVM C={0,1; 1; 10}, CRC λ={0,1; 1,35; 10}; chọn mean CV Rank-1 cao nhất, hòa chọn tham số nhỏ nhất. SVM chọn C=1; CRC chọn λ=0,1. Tất cả artifact đã khóa trước đánh giá session2.

| Model | Mean CV Rank-1 | CPU median (ms/ảnh) | CPU P95 (ms/ảnh) | Fit final (s) |
|---|---:|---:|---:|---:|
| knn | 99.9833% | 10.9987 | 20.2279 | 0.007858 |
| svm | 100.0000% | 3.7102 | 5.5744 | 45.583415 |
| crc | 100.0000% | 14.7064 | 20.8129 | 0.076052 |

CPU Ryzen 7 5800H, một luồng, 6.000 probe, model nạp sẵn. Tổng thời gian gồm đọc ROI, CompCode và score cho 600 lớp, chưa gồm xếp hạng/UI. Fit final không gồm trích feature/CV. Không cộng các median thành phần để suy ra median tổng.

CI Rank-1 từng run nằm trong bảng CSV, bootstrap theo Palm ID, 1.000 lần, seed=42. Paired CRC − SVM là +0,9000 điểm phần trăm, CI 95% [0,4000; 1,5000]. Xem [bảng paired](tables/paired_rank1_differences.csv) và [lưới CV](tables/tongji_cv_grid_summary.csv).

Classical source SHA-256 khi chạy: 4e324e0b8840cb473dcda4cfcf75c3344b21f64d03360a2efe0bb502ad3b05bc. Mã hiện tại đã được dọn sau benchmark; snapshot gốc và các khóa được giữ nguyên tại máy. Môi trường được lưu trong [experiments/environments](../experiments/environments/README.md).

CompCode là bản Python paper-inspired, chưa xác minh parity MATLAB. Tham khảo [trang tác giả Tongji](https://cslinzhang.github.io/ContactlessPalm/) và [bài báo gốc](https://cslinzhang.github.io/ContactlessPalm/contactlesspalm.pdf).
