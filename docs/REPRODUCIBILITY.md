# Tái lập thí nghiệm và kiểm soát thay đổi

## Quy trình tối thiểu

1. `pip install -e '.[test,demo]'`, cài thêm `[cnn]` nếu cần.
2. Dữ liệu Official ROI dưới `data/raw/session1` và `data/raw/session2`. Dữ liệu không đi kèm gói.
3. `palmprint prepare --data data/raw` tạo manifest, SHA256, folds; lệnh strict mặc định bắt buộc đúng 600 Palm ID và mỗi Palm ID có 10 ảnh/session.
4. `palmprint eda` sinh histogram chất lượng/đếm lớp từ manifest, **không xuất ảnh sinh trắc học montage công khai**.
5. `palmprint features` lưu cache `.npz` có fingerprint manifest và cấu hình; không fit một biến đổi học từ test.
6. `palmprint cv --model <knn|svm|crc>` chạy 5 folds session1 và lưu `outputs/cv/compcode_<model>.csv` + `_selection.json`.
7. `palmprint train --model <...>` huấn luyện lại trên session1, tạo `artifacts/<model>.joblib` và `<model>_freeze.json`; thực hiện cho cả ba mô hình.
8. `palmprint lock` khóa đồng thời ba mô hình và hồ sơ thực nghiệm trước lần truy cập test.
9. `palmprint evaluate --model <...>` đánh giá từng mô hình trên session2; từ chối ghi đè kết quả test.
10. `palmprint errors --model <...>`, `palmprint benchmark --model <...>`, `palmprint report` xuất số liệu và biểu đồ.
11. `streamlit run app/main.py` nạp artifact đã được lưu. Với các mô hình cổ điển, cần chạy `lock` trước khi dùng ảnh test; giao diện chưa có kiểm tra khóa giao thức ở mức app. Với CNN, cấu hình và seed cần được đăng ký trước lần đánh giá session2.

## Metadata

- `data/metadata/manifest.csv`: sample UID, Palm ID, session, path tương đối, SHA256, quality proxies.
- `data/splits/fold_assignments.csv`: chỉ lưu các ảnh session1 và mã fold; session2 thuộc tập test, không được đưa vào danh sách fold phát triển.
- `outputs/cache/*.npz`: feature cache có signature; không tái dùng nếu manifest hoặc config khác.
- `outputs/cv/*_selection.json`: tham số và mean/std CV.
- `artifacts/*_freeze.json`: fingerprint artifact/manifest/feature, danh sách class.
- `outputs/runs/<model>/predictions.csv`: sample UID, nhãn thật, dự đoán, thứ hạng nhãn thật và Top-5 Palm ID.
- `outputs/runs/<model>/metrics.json`: Rank-1/5, F1, CI, thời gian tính điểm.
- `outputs/runs/<model>/confusion_matrix.npy`: ma trận đầy đủ.
- `reports/tables/model_comparison.csv`: tổng hợp tự động, không sửa tay.

## So sánh công bằng

- M1–M3 dùng chính xác cùng CompCode feature và CV folds.
- M4 dùng cùng Official ROI nhưng tensor 224×224 và learned features, do đó chỉ so toàn pipeline.
- SVM/CRC có giá trị tham số chọn bằng CV, không đặt bằng cách tối ưu trên session2.
- Mặc định không dùng `StandardScaler` riêng cho SVM trong phép so sánh chính.
- E3a: chạy `cv --model crc --representation global` (6 chiều vs 600 chiều) trên session1.
- E3b: chạy `cv --model svm --representation hog` vs CompCode SVM (chỉ session1).
- E6: `palmprint robustness --model <...>` chạy trên session2 sau main result; cấu hình perturbation cố định trong mã nguồn trước test, không chọn lại model theo kết quả.

## Tại sao chưa gọi CR_CompCode nguyên bản?

Mã nguồn chính thức của tác giả: <https://cslinzhang.github.io/ContactlessPalm/CR_CompCode.rar>. Repo này có bản triển khai riêng bằng Python (Gabor kernel, quantization, histogram normalization, ridge residual). Cần kiểm tra parity trên cùng ROI: 6 response maps, orientation map, block-wise vector, residual và class ranking với nguồn MATLAB. Nếu khác, báo cáo là **paper-inspired**, không sao chép con số 98,78% từ paper thành kết quả nhóm.

## Bảo mật và dữ liệu

- **Không git add ảnh sinh trắc học hoặc checkpoints từ dataset**; `.gitignore` chặn các đường dẫn mặc định.
- Các ví dụ demo lấy từ nguồn hợp pháp và trình bày đúng phạm vi.
- Không dùng đường dẫn thật của máy thành hướng dẫn bắt buộc; có thể cấu hình `--data`.

## Xử lý sự cố

- `Strict Tongji mode`: kiểm tra đủ 600 ID × 10 ảnh mỗi session và tên 5 chữ số.
- `Cross-session identical hashes`: điều tra bản sao ảnh; không được im lặng xoá/chia lại.
- `Dataset changed after freeze`: manifest/config/artifact có thay đổi; phải sử dụng workspace mới nếu làm nghiên cứu khác.
- `Test already evaluated`: kết quả test đã khóa; không xóa/thay/ghi đè để tune.
- `MemoryError CRC`: giảm batch trên inference, hạn chế đồng thời CV; thuật toán dùng factorization m×m.
- CNN `weights=imagenet1k_v1` có thể cần tải weights lần đầu; nếu không có internet, `--weights none` để kiểm thử **không tương đương pretrained**.
- `ModuleNotFoundError streamlit`: cài `pip install -e '.[demo]'`.


## Source sau benchmark

Đợt dọn repository chỉ chuẩn hóa style/import, gom CNN runtime và cập nhật mô tả. Thuật toán, config, trọng số, raw metrics và bảng benchmark được giữ nguyên. Hash source hiện tại khác hash đã khóa khi chạy; không sửa freeze/protocol_lock để bỏ qua kiểm tra này.

- Muốn kiểm tra lịch sử chính xác: dùng source_snapshot cùng môi trường của đợt chạy, được giữ cục bộ trong outputs/tongji_benchmark/ và outputs/cnn_experiments/.
- Muốn chạy một nghiên cứu mới bằng source hiện tại: tạo workspace mới, chuẩn bị manifest, CV/train/lock và đánh giá theo giao thức mới. Không ghi đè hoặc gọi lại test của workspace đã có kết quả.
- Repo GitHub chỉ chứa mã, config, tài liệu, bảng tổng hợp và hai hình nghiên cứu đã chọn. Checkpoint, gallery, manifest, predictions và log đầy đủ nằm tại máy.
- Dependency snapshot của các lần chạy nằm trong [experiments/environments](../experiments/environments/README.md). Đây là hồ sơ phiên bản Windows/Python 3.11, không phải lockfile dùng chung mọi hệ điều hành.
