# Hồ sơ thí nghiệm

registry.yaml ghi thiết kế E0–E6. Trạng thái thực tế và số liệu nằm trong [benchmark](../reports/TONGJI_BENCHMARK.md) và [báo cáo CNN](../reports/CNN_EXPERIMENTS.md), không suy ra từ registry rằng mọi ablation đã chạy.

Đã chạy M1–M3, ResNet18 ba seed và hai biến thể cosine từ cùng checkpoint. Cosine được bổ sung sau benchmark logits, là đánh giá hậu nghiệm; không sửa đăng ký cũ thành một kế hoạch được chốt từ trước. Chưa chạy các ablation/robustness thật chỉ vì đã có code hoặc test synthetic.

[Snapshot môi trường](environments/README.md) được đưa vào repo. Source snapshot, protocol, artifacts và raw evidence đầy đủ nằm dưới outputs/ và artifacts/, được lưu cùng workspace trong repo private. Git repo được tạo sau các thực nghiệm; commit hiện tại không thay thế hash source đã lưu khi chạy.
