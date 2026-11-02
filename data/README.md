# Tongji dataset layout

Official ROI từ trang tác giả: https://cslinzhang.github.io/ContactlessPalm/

Giải nén `Official ROI` vào `data/raw/session1` và `data/raw/session2`, mỗi session dự kiến 6000 ảnh, mỗi 10 ảnh liên tiếp là một Palm ID. Không bỏ Original vào cùng thư mục `data/raw`.

Tập ảnh nặng `data/raw/` được lưu trữ cục bộ và loại khỏi git tracking để tối ưu dung lượng kho lưu trữ. Người dùng tải bộ dữ liệu từ liên kết tác giả ở trên và giải nén vào `data/raw/` theo cấu trúc quy định.

`data/metadata/local_dataset.json` được tạo từ `prepare` chứa đường dẫn dataset của máy hiện tại. Nếu di chuyển ZIP sau chạy, chuẩn bị lại từ dataset gốc bằng `palmprint prepare` ở workspace mới.
