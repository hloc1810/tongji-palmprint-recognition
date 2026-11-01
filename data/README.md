# Tongji dataset layout

Official ROI từ trang tác giả: https://cslinzhang.github.io/ContactlessPalm/

Giải nén `Official ROI` vào `data/raw/session1` và `data/raw/session2`, mỗi session dự kiến 6000 ảnh, mỗi 10 ảnh liên tiếp là một Palm ID. Không bỏ Original vào cùng thư mục `data/raw`.

Repo private hiện lưu cả `data/raw`, `artifacts` và `outputs/runs` theo yêu cầu của chủ dự án. `.gitignore` ở gốc đã được bỏ; không công khai bản lưu này khi chưa rà soát quyền sử dụng dữ liệu.

`data/metadata/local_dataset.json` được tạo từ `prepare` chứa đường dẫn dataset của máy hiện tại. Nếu di chuyển ZIP sau chạy, chuẩn bị lại từ dataset gốc bằng `palmprint prepare` ở workspace mới.
