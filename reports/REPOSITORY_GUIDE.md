# Cấu trúc repository Tongji Palmprint Recognition

Repository tổ chức mã theo các bước của bài toán nhận dạng closed-set 1:N.
Hai phiên Tongji có cùng 600 Palm ID; Session 1 dùng để chọn cấu hình/học model
và tạo gallery, Session 2 dùng để đánh giá cuối. Bàn tay trái và phải là hai
Palm ID riêng. Quy tắc cụ thể của từng thực nghiệm nằm trong báo cáo tương ứng.

| Vị trí | Vai trò |
|---|---|
| `src/palmprint/data/` | Đọc và kiểm tra ROI, Palm ID, manifest và các fold Session 1. |
| `src/palmprint/features/` | CompCode, các đặc trưng so sánh và chọn đặc trưng cho PalmNet. |
| `src/palmprint/models/` | 1-NN, Linear SVM, CRC_RLS, ResNet và matching PalmNet sparse. |
| `src/palmprint/training/` | Cross-validation, huấn luyện và đóng băng; M5 có luồng chạy độc lập. |
| `src/palmprint/evaluation/` | Ranking theo lớp, Rank-k, Macro-F1, CMC và bootstrap theo Palm ID. |
| `src/palmprint/reporting/` | Tổng hợp và xác minh bằng chứng của các thực nghiệm hiện có. |
| `configs/` | Cấu hình dataset, CompCode, classifier, ResNet và cấu hình tham chiếu M5. |
| `scripts/` | Entry point cho các thí nghiệm phân tích và benchmark M5. |
| `app/` và `Tongji_Palmprint_UI.html` | Demo Streamlit và frontend/API FastAPI cho các model CompCode. |
| `data/metadata/` | Manifest 12.000 ảnh và metadata kiểm tra dữ liệu. |
| `data/raw/` | Official ROI trên máy cục bộ; ảnh gốc không đưa lên Git. |
| `artifacts/` | Model và hiện vật huấn luyện; checkpoint bộ trích xuất M5 nằm trong `pretrained/`, gallery và classifier M5 lớn được giữ cục bộ. |
| `outputs/` | Metrics, dự đoán và kết quả thực nghiệm; cache và ma trận điểm M5 được giữ cục bộ. |
| `reports/`, `notebooks/`, `experiments/` | Báo cáo, hình vẽ, khám phá dữ liệu và mô tả thí nghiệm. |
| `tests/`, `.github/workflows/` | Kiểm thử và GitHub Actions trên Python 3.11/3.12. |

Entry point chính là `palmprint` cho các pipeline CompCode M1–M3,
`palmprint-cnn` cho ResNet M4, và `palmprint-palmnet` hoặc
`python scripts/run_palmnet_m5.py` cho M5. M5 hiện là thí nghiệm độc lập:
API và giao diện nhận dạng hiện tại vẫn nạp ba model CompCode M1–M3; kết quả
M5 đã được bổ sung vào tài liệu và bảng benchmark.
Để cài các dependency riêng của API, dùng `python -m pip install -e ".[api]"`.

M5 dùng Python port PCA → Gabor của tác giả PalmNet, không học bằng
backpropagation. Sau binary hashing và histogram theo block, 1-NN đối sánh
descriptor sparse đầy đủ. Các nhánh SVM và CRC_RLS dùng chung phép chọn 1.024
cột từ Session 1; 1-NN cùng biểu diễn là đối chứng classifier. Xem
[báo cáo M5](PALMNET_M5.md) để đọc giao thức, kết quả, chi phí và lệnh chạy.

Để tái lập, tải Official ROI theo [hướng dẫn dữ liệu](../data/README.md), cài
dependency và mã nguồn PalmNet như trong báo cáo M5. Không thay đổi cấu hình
dựa trên kết quả Session 2; các lựa chọn mới cần được đánh giá trên Session 1.
