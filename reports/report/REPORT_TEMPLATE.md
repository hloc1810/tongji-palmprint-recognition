# Đề cương báo cáo kỹ thuật (20–30 trang)

> Không điền số liệu tự chế. Mọi bảng kết quả phải lấy từ `outputs/runs/**/metrics.json` và `reports/tables/`.

## Chương 1 — Giới thiệu (3 trang)
- Palmprint biometric identification 1:N; closed-set và giới hạn.
- Câu hỏi nghiên cứu RQ1–RQ4; đóng góp phương pháp, phạm vi, bài báo nguồn.

## Chương 2 — Cơ sở lý thuyết (5 trang)
- Tongji ROI; Gabor/Competitive Code histogram; biểu diễn theo block.
- k-NN, Linear SVM, CRC_RLS: công thức đúng với source code.
- CNN ResNet18: transfer learning, head training / layer4 fine-tuning.
- Nguồn: Duda giáo trình cho classical; CompCode/CRC paper Tongji; ResNet paper gốc.

## Chương 3 — Dataset và phương pháp (5 trang)
- 300 người / 600 palms / 12k ảnh/2 session; nhãn filename.
- Audit số lượng lớp, lỗi/duplicates, EDA brightness/contrast/blur.
- Cross-session protocol, 5-fold validation nội bộ, lịch sử pilot session2.

## Chương 4 — Thực nghiệm (6 trang)
- Môi trường phần cứng, thư viện, versions.
- Hyperparameter grids và fold assignments.
- Bảng model comparison, CMC, confusion matrix, CI, speed.
- Ablation global vs local histogram; HOG vs CompCode dưới SVM cố định.
- Training curves CNN, seed-to-seed variability.

## Chương 5 — Phân tích (5 trang)
- Top confusion pairs, 10–15 mẫu sai + 5 mẫu đúng, quality associations.
- So sánh Rank-1, Rank-5, Macro-F1, CPU latency, artifact size.
- Những giới hạn ngoại suy, sự phụ thuộc theo Palm ID và Person ID không biết.

## Chương 6 — Kết luận (2 trang)
- Kết luận theo RQ1–RQ4 có số thực nghiệm; điều chưa chứng minh.
- Demo và khả năng phát triển trong tương lai.

## Phụ lục, trích dẫn
- Link dataset và bằng chứng nguồn được duyệt.
- Link GitHub, câu lệnh reproduce, bảng phân công, bibliography.
