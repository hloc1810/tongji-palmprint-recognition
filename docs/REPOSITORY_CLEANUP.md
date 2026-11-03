# Biên bản tổ chức repository

Ngày 03/11/2026, múi giờ Asia/Saigon. Phạm vi: tổ chức mã/hiện vật, tài liệu và
kiểm tra phần mềm; không huấn luyện lại mô hình Tongji hoặc đổi số liệu nghiên cứu.

## Thay đổi cấu trúc

| Trước | Sau | Lý do |
|---|---|---|
| HTML ở gốc repository | `app/static/index.html` | Gom frontend cùng ứng dụng |
| Đề bài PDF ở gốc | `docs/references/project_assignment.pdf` | Tập trung tài liệu tham chiếu |
| Script vẽ ROI trong `outputs/` | `scripts/render_roi_visual_review.py` | Tách code duy trì khỏi kết quả |
| Hai wrapper chạy M5 | Giữ `scripts/run_palmnet_m5.py` và CLI | Loại wrapper gốc trùng chức năng |
| Nhiều hướng dẫn rải trong README | README tổng quan, docs phương pháp/tái lập/bảo trì | Tách vai trò tài liệu |

Backend vẫn phục vụ URL `/Tongji_Palmprint_UI.html` để các liên kết cũ hoạt động.
Việc chuyển HTML không thay nội dung giao diện. Thư viện `src/palmprint/`, các
cấu hình khoa học và thuật toán đã khóa giữ nguyên.

## Phần dư được loại khỏi cây làm việc

- Fixture pytest cũ, workspace kiểm thử synthetic và các thư mục `*_tmp`.
- Hai bản chạy thử ROI trước bản final và thư mục test ROI.
- ZIP, checkout, XML và script xuất bản thuộc đợt dọn repository cũ.
- Screenshot before/after, bảng/báo cáo bản sao và script sửa bảng embedding cũ.

Tổng cộng **12.135 file**, khoảng **1,67 GiB**, được chuyển khỏi vị trí làm việc
sang kho khôi phục `.local/archive/obsolete/`. Đây là dọn cấu trúc có thể khôi phục,
**không phải số dung lượng ổ đĩa đã giải phóng**. Các file tracked lỗi thời được
loại khỏi phiên bản Git mới; lịch sử commit vẫn giữ các phiên bản trước đó.

Kết quả PCA migration chưa công bố, hình thử nghiệm NumPy và chẩn đoán M5 được
lưu trong `.local/archive/`, không bị xóa hoặc đưa vào kết quả chính. Junction
`node_modules` trỏ tới dependency dùng chung bên ngoài workspace được giữ nguyên;
không can thiệp dữ liệu đích đó. `.local/`, test mới và `node_modules` được ignore.

## Bằng chứng được bảo toàn

Đã chụp SHA256 của **568 file** trước khi tổ chức lại, gồm mã thư viện, cấu hình,
model/freeze, metadata/split và bằng chứng kết quả cần giữ. Không chỉnh metric,
prediction, score matrix, protocol hoặc source snapshot để khớp với tài liệu mới.
Các dataset ảnh và gallery/cache PalmNet nhiều GiB giữ nguyên vị trí.

Danh sách từng file và kế hoạch khôi phục nằm trong `.local/cleanup/` trên máy
thực hiện. [Biên bản kiểm tra phát hành](../validation/README.md) ghi kết quả
kiểm thử và xác minh sau thay đổi; không dùng log kiểm thử cũ làm bằng chứng
cho phiên bản mã mới.
