# Hồ sơ thí nghiệm

[registry.yaml](registry.yaml) ghi thiết kế E0–E6 ban đầu; không sửa ngược để coi
những mở rộng làm sau là đã đăng ký trước. Trạng thái và số liệu nằm trong
[benchmark](../reports/TONGJI_BENCHMARK.md), [CNN](../reports/CNN_EXPERIMENTS.md)
và [ablation E3](../outputs/feature_ablation/findings.md).

Đã thực hiện:
- **M1–M3**: 1-NN, Linear SVM và CRC_RLS trên đặc trưng CompCode 600-D (5-fold CV và Session 2 test).
- **M4**: ResNet18 ba seed (17, 29, 43) và hai chiến lược so khớp Cosine Embedding hậu nghiệm (Nearest & Centroid).
- **E3**: Thực nghiệm bóc tách biểu diễn đặc trưng (Raw Grayscale 16.384-D vs PCA 256-D vs CompCode 600-D trên 1-NN) đánh giá cả 5-fold CV nội phiên và kiểm thử chéo phiên trên bộ dữ liệu Tongji thật tại [outputs/feature_ablation/](../outputs/feature_ablation/findings.md).
- **M5 bổ sung**: PCA-Gabor PalmNet với descriptor sparse đầy đủ và các nhánh
  chọn đặc trưng, theo [giao thức M5](../reports/PALMNET_M5.md).
- **ROI pilot bổ sung**: 10 Palm ID, 200 ảnh ghép cặp, giữ nguyên CompCode + 1-NN,
  theo [giao thức ROI](../reports/ROI_COMPARISON.md).

[Snapshot môi trường](environments/README.md) được lưu theo lần chạy.
Source snapshot, protocol và bằng chứng nằm trong `outputs/` và `artifacts/`.
Dataset, cache/score/model lớn không có đầy đủ trên GitHub; xem
[hướng dẫn tái lập](../docs/REPRODUCIBILITY.md). Commit mã hiện tại không thay thế
hash source đã lưu khi thực nghiệm. Kết quả thử chưa công bố được giữ cục bộ
trong `.local/archive/`, không trộn vào các bảng kết quả chính.
