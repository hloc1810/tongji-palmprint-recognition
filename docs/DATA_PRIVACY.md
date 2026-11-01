# Dữ liệu, quyền riêng tư và giới hạn suy luận

Dữ liệu Tongji chứa hình ảnh sinh trắc học cá nhân. Nhóm chỉ sử dụng theo điều khoản và nguồn công bố của tác giả; tài liệu giảng viên yêu cầu cung cấp link nguồn và được duyệt trước khi thực hiện.

- Trang nguồn: https://cslinzhang.github.io/ContactlessPalm/
- Hãy tải **Official ROI Images**; `Original Images` chỉ phục vụ khảo sát ROI, không phải tập ảnh độc lập.
- Repo hiện là bản lưu private toàn workspace theo yêu cầu của chủ dự án, không dùng `.gitignore` ở gốc. Không đưa ảnh ROI, checkpoint hoặc gallery lên GitHub công khai khi chưa được phép; cần rà soát nội dung trước khi đổi chế độ truy cập.
- Không gán `Person ID`, tay trái/phải từ Palm ID nếu không có metadata chính thức.
- Demo chỉ nhận ROI cùng miền dữ liệu, **không** bảo đảm ảnh smartphone, ảnh bàn tay trọn vẹn hoặc người ngoài tập đăng ký.
- App nhận ROI upload vào file tạm, không nhập vào tập huấn luyện; triển khai public cần bổ sung chính sách xoá file tạm, HTTPS và quản lý truy cập trước khi dùng thật.
- `blur_proxy` (Laplacian variance) chỉ là proxy thống kê; mức độ mờ/ảnh kém chất lượng ảnh hưởng đến nhận dạng cần được kiểm chứng.
