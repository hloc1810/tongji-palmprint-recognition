# Quy tắc bảo trì

## Vị trí và trách nhiệm

- Logic tái sử dụng đặt trong `src/palmprint/` theo trách nhiệm module.
- Entry point phân tích đang duy trì đặt trong `scripts/`; không thêm wrapper trùng ở gốc.
- Giao diện đặt trong `app/`, tài sản frontend trong `app/static/`.
- Hướng dẫn đặt trong `docs/`, kết quả và diễn giải đặt trong `reports/`.
- Output mới, test tạm và bản khôi phục đặt trong `.local/`, không đưa lên Git.
- Bằng chứng công bố đặt trong `outputs/<experiment>/`; ghi nguồn và protocol.

Snapshot trong `outputs/*/source_snapshot/` là hồ sơ lịch sử, không chỉnh đồng
bộ với `src/`. Trước khi xóa file kết quả, kiểm tra tham chiếu của code, báo cáo,
freeze và protocol. Cache PalmNet lớn còn được dùng để tiếp tục/tái kiểm tra;
không xóa chỉ vì tên thư mục là `cache`.

## Trước khi commit

```powershell
git status --short
git diff --check
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q -ra -p no:cacheprovider --basetemp .local/tests
python scripts/run_smoke.py
```

Chạy kiểm thử phù hợp với thay đổi; CI kiểm tra Python 3.11 và 3.12. Không đưa
dataset, môi trường ảo, file PID, log phát triển hoặc test fixture sinh tự động
vào commit. Thêm file cụ thể vào staging sau khi xem diff.

## Thay đổi khoa học

Đổi segmentation, feature, classifier hoặc cách tính metric cần mô tả ảnh hưởng
và thực nghiệm mới trong workspace riêng. Không viết lại checksum/khóa của run
cũ để hợp thức hóa mã mới. Với CSV/JSON được hash theo byte, kiểm tra cả nội dung
trong Git index vì Git có thể chuẩn hóa xuống dòng.

Giữ kết luận đúng mức: không đồng nhất std với CI, crop vượt kiểm tra với crop
đúng giải phẫu, hoặc không có ý nghĩa thống kê với bằng chứng tương đương.
