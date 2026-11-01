# M4-E2: ResNet18 Embedding Matching — Tongji Contactless Palmprint

## Mục tiêu

Bổ sung phương pháp nhận dạng 1:N dựa trên embedding cho ResNet18 đã huấn luyện. Phương pháp
**không thay đổi trọng số ResNet18**, không huấn luyện metric loss mới và không ghi đè
kết quả M4-E1 (600-class classification) hiện có.

Giao thức:

- Input: ảnh Official ROI 128×128 grayscale; dùng lại preprocessing evaluation của project.
- Backbone: ResNet18, thay tầng `fc` bằng `Identity` **sau khi** nạp state_dict của checkpoint.
- Output: vector 512 chiều trước tầng `fc`; L2-normalize vector.
- Gallery: 6.000 ảnh Session 1 (600 lớp, 10 ảnh/lớp).
- Query: 6.000 ảnh Session 2 (600 lớp, 10 ảnh/lớp).
- Ranking: cosine similarity với hai chiến lược nearest-template hoặc centroid.
- Metrics: Rank-1/Rank-5/Macro-F1, CMC và confusion matrix 600×600.

Định nghĩa:

\[
 z(x)=\frac{f_\theta(x)}{\|f_\theta(x)\|_2},\qquad
 s(x,g)=z(x)^\top z(g).
\]

**Nearest template:**

\[
 S_k(x)=\max_{g\in\mathcal G_k}s(x,g),\qquad
 \hat{k}=\arg\max_k S_k(x).
\]

**Centroid:**

\[
 c_k=\operatorname{L2Norm}\left(\frac{1}{|\mathcal G_k|}\sum_{g\in\mathcal G_k}z(g)\right),
 \qquad S_k(x)=z(x)^\top c_k.
\]

Hàm `ranking` của project quy định **score càng nhỏ càng tốt**, vì vậy code chuyển
similarity sang `-cosine_similarity` trước khi xếp hạng. Điều này không làm thay đổi
thứ tự kết quả.

## Chạy với checkpoint hiện có

Cần có 3 nhóm file tại project đang chạy:

1. `data/metadata/manifest.csv` và `data/metadata/local_dataset.json`.
2. Ảnh ROI thật ở `data_root` được trỏ tới trong `local_dataset.json`.
3. `artifacts/resnet18_seed17.pth` và `artifacts/resnet18_seed17_freeze.json`
   (lặp tương tự cho seed 29/43).

Chạy trong thư mục project sau khi `pip install -e '.[cnn]'`:

```bash
# Xây dựng gallery từ Session 1; không đọc ảnh Session 2:
palmprint-cnn --workspace . embedding-gallery --seed 17 --device cpu

# Đo trực tiếp cả nearest-template và centroid trên Session 2:
palmprint-cnn --workspace . embedding-evaluate --seed 17 --strategy both --device cpu

# Nhận dạng một ảnh Official ROI 128x128:
palmprint-cnn --workspace . embedding-predict --seed 17 --strategy nearest --image /path/to/roi.bmp
```

Để báo cáo nhiều seed, chạy `embedding-evaluate` lần lượt với 17, 29, 43 và thống kê
mean ± sample SD của Rank-1/Rank-5/Macro-F1. Không lựa chọn seed theo Session 2.

**Các file xuất:** `artifacts/resnet18_seed17_gallery512.npz`,
`outputs/runs/resnet18_seed17_embedding_nearest/` và
`outputs/runs/resnet18_seed17_embedding_centroid/`.

## Kết quả hiện tại và giới hạn

Đã chạy cả ba seed 17, 29, 43 trên Tongji thật. Nearest Cosine đạt mean Rank-1 73,7444% ± 0,6241 điểm phần trăm; Centroid đạt 68,8278% ± 0,7558. Số liệu đầy đủ, CI từng seed và timing nằm trong [bảng benchmark](../reports/TONGJI_BENCHMARK.md).

- Đây là đánh giá hậu nghiệm sau benchmark logits/classical, không phải test set chưa từng quan sát. Không chọn seed hoặc chiến lược theo session2.
- Dùng cùng checkpoint cross-entropy đã có; không train contrastive/triplet/ArcFace loss.
- Centroid là trung bình các embedding đã L2-normalize rồi chuẩn hóa lại. Cả hai chiến lược trả 600 score lớp; Top-5 đếm Palm ID khác nhau.
- Gallery cache gắn checkpoint hash, manifest hash, kích thước ảnh và thứ tự UID/Palm ID. Không phân phối ảnh, gallery hoặc checkpoint trong repo.
- File metrics gốc có warning kế thừa về historical pilots; chưa có bằng chứng xác nhận pilot HOG/120 ID. Điều đã biết là session2 đã được đánh giá trước khi bổ sung embedding.
- Thời gian batch chia số ảnh khác median/P95 từng ảnh. Chưa đo latency từng ảnh cho embedding.
- Chạy lại trong workspace mới nếu cần nghiên cứu khác; evaluator từ chối ghi đè kết quả hiện có.
