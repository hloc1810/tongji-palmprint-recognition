# Danh mục báo cáo

| Báo cáo | Phạm vi | Vai trò |
|---|---|---|
| [EDA](EDA.md) | Dataset và chất lượng ảnh | Mô tả phân bố và đặc điểm dữ liệu |
| [Benchmark M1–M4](TONGJI_BENCHMARK.md) | 600 lớp, 6.000 probe/run | Bảng so sánh classical/CNN/embedding |
| [CNN](CNN_EXPERIMENTS.md) | 3 seed, logits và cosine | Thiết kế, sai số số học, giới hạn hậu nghiệm |
| [PalmNet M5](PALMNET_M5.md) | 600 lớp, descriptor sparse đầy đủ | Giao thức, kết quả, chi phí và nguồn tác giả |
| [Custom ROI](ROI_COMPARISON.md) | 10 lớp, 200 ảnh ghép cặp | Crop, 1-NN, tốc độ và lỗi kẽ ngón |
| [Ablation E3](../outputs/feature_ablation/findings.md) | Raw/PCA/CompCode + 1-NN | Tách ảnh hưởng biểu diễn |
| [Robustness E6](../outputs/robustness_e6/findings.md) | Model khóa và biến đổi ảnh | Đánh giá độ nhạy trên Session 2 |
| [Spatial E3a](../outputs/spatial_ablation_e3a/findings.md) | CV Session 1 | Global và spatial histogram |
| [Kiến trúc](REPOSITORY_GUIDE.md) | Mã nguồn và hiện vật | Điều hướng repository |

`tables/` và `figures/` chứa bảng/hình dẫn xuất. Prediction, score, metric và
protocol gốc nằm trong [outputs](../outputs/README.md). Không cộng hoặc so trực
tiếp số liệu có khác mẫu số, tập lớp hay phạm vi timing.
