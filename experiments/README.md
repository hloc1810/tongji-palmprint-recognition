# Hồ sơ thí nghiệm

registry.yaml ghi thiết kế E0–E6. Trạng thái thực tế và số liệu nằm trong [benchmark](../reports/TONGJI_BENCHMARK.md), [báo cáo CNN](../reports/CNN_EXPERIMENTS.md), và [nghiên cứu bóc tách đặc trưng E3](../outputs/feature_ablation/findings.md).

Đã thực hiện:
- **M1–M3**: 1-NN, Linear SVM và CRC_RLS trên đặc trưng CompCode 600-D (5-fold CV và Session 2 test).
- **M4**: ResNet18 ba seed (17, 29, 43) và hai chiến lược so khớp Cosine Embedding hậu nghiệm (Nearest & Centroid).
- **E3**: Thực nghiệm bóc tách biểu diễn đặc trưng (Raw Grayscale 16.384-D vs PCA 256-D vs CompCode 600-D trên 1-NN) đánh giá cả 5-fold CV nội phiên và kiểm thử chéo phiên trên bộ dữ liệu Tongji thật tại [outputs/feature_ablation/](../outputs/feature_ablation/findings.md).

[Snapshot môi trường](environments/README.md) được đưa vào repo. Source snapshot, protocol, artifacts và raw evidence đầy đủ nằm dưới outputs/ và artifacts/, được lưu cùng workspace trong repo private. Git repo được tạo sau các thực nghiệm; commit hiện tại không thay thế hash source đã lưu khi chạy.
