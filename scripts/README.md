# Entry point đang duy trì

Chạy từ thư mục gốc sau `pip install -e .`. Logic chung nằm trong `src/palmprint`.

| Script | Tác vụ | Dữ liệu/phạm vi |
|---|---|---|
| `run_smoke.py` | Kiểm tra tích hợp | Synthetic; không phải kết quả nghiên cứu |
| `run_roi_comparison.py` | So sánh Official/Custom ROI | 10 ID, hai bộ ảnh, output mới |
| `render_roi_visual_review.py` | Vẽ ví dụ đã gán nhãn ROI | Pilot đã lưu, không chấm đúng/sai tự động |
| `run_palmnet_m5.py` | Điều phối M5 | Port tác giả, gallery/cache lớn |
| `summarize_palmnet_m5.py` | Tổng hợp kết quả M5 | Bằng chứng M5 đã tồn tại |
| `run_feature_ablation.py` | Raw/PCA/CompCode + 1-NN | E3, Session 1 CV và cross-session |
| `run_paired_bootstrap_e3.py` | Bootstrap ghép cặp | Kết quả E3 đã tồn tại |
| `run_spatial_ablation_e3a.py` | Global/spatial CompCode | Session 1 CV |
| `run_robustness_e6.py` | Biến đổi probe với model khóa | Phân tích hậu nghiệm |
| `audit_research_correctness.py` | Audit input và sanity feature | Cần workspace mới vì có bước prepare |
| `generate_report_visuals.py` | Vẽ hình báo cáo | Bằng chứng đã lưu |
| `generate_dimensionality_reduction_visuals.py` | Vẽ PCA | Dữ liệu/cache theo script |

Các script phân tích lịch sử có thể ghi vào output cố định và không phải script
nào cũng nhận `--workspace`. Đọc phần cấu hình trước khi chạy; dùng một bản sao
workspace riêng cho nghiên cứu mới. Không chạy tất cả script trên kết quả khóa
chỉ để kiểm tra cài đặt. Xem [hướng dẫn tái lập](../docs/REPRODUCIBILITY.md).
