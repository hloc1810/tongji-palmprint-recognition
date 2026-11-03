# PCA tự triển khai cho nhận dạng lòng bàn tay

Lớp `PCA` trong `src/palmprint/features/pca.py` tự thực hiện các bước PCA,
không gọi `sklearn.decomposition.PCA`. NumPy vẫn được dùng cho phép nhân ma
trận, QR và SVD; đây không phải là triển khai đại số tuyến tính thuần Python.

Với ma trận huấn luyện X gồm n ảnh đã làm phẳng, mỗi hàng là một ảnh:

1. Tính ảnh trung bình μ theo từng pixel và đặt A = X − μ.
2. Tạo ma trận ngẫu nhiên Ω với seed cố định; QR của AΩ tạo cơ sở Q gần
   không gian cột của A. Bốn vòng lặp lũy thừa, có QR ổn định ở mỗi bước,
   cải thiện độ chính xác của các hướng có phương sai lớn.
3. Phân tích SVD ma trận nhỏ B = QᵀA. Chọn k vector kỳ dị phải đầu tiên
   làm các hàng của `components_`; đây là xấp xỉ các hướng chính của A.
4. Chiếu ảnh mới bằng Z = (X − μ) components_ᵀ. Phương sai của thành phần i
   được ước lượng bằng σᵢ²/(n−1); tỉ lệ phương sai dùng tổng phương sai của
   toàn bộ A làm mẫu số. Có thể tái tạo gần đúng bằng Z components_ + μ.

Trong E3, ảnh ROI xám 128×128 được làm phẳng và chuẩn hóa L2, sau đó PCA
giảm 16.384 chiều xuống 256 chiều; vector chiếu tiếp tục chuẩn hóa L2 trước
khi so khớp 1-NN. PCA chỉ fit trên train của từng fold, hoặc trên Session 1
trong đánh giá cross-session; Session 2 chỉ được transform bằng μ và các trục
đã học. Việc giảm chiều là không giám sát, không bảo đảm các hướng giữ lại
có khả năng phân biệt 600 Palm ID tốt nhất.

Randomized SVD cho kết quả xấp xỉ và có thể khác triển khai scikit-learn dù
cùng seed. Các bảng và hình E3 đã lưu là kết quả cũ, không phải kết quả đã
xác nhận của phiên bản này. Chạy lại `python scripts/run_feature_ablation.py
--workspace <workspace>` trong môi trường dự án và đánh giá Rank-1, Rank-5;
script ghi vào `outputs/feature_ablation` của workspace, vì vậy cần giữ bản
sao kết quả cũ hoặc dùng workspace riêng nếu muốn bảo toàn chúng. PCA-Gabor
PalmNet M5 là pipeline mã tác giả riêng và không được thay đổi ở đây.
