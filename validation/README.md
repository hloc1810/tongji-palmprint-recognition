# Kiểm tra phiên bản tổ chức lại repository

Ngày 03/11/2026, Asia/Saigon. Kiểm tra này đánh giá phần mềm và bảo toàn hiện vật,
không huấn luyện lại mô hình Tongji hoặc tạo thêm số liệu nhận dạng trên Tongji.

## Lệnh kiểm tra

```powershell
python -m ruff check src app scripts tests
python -m ruff format --check src app scripts tests
python -m pytest -q -ra -p no:cacheprovider --basetemp .local/tests
python scripts/run_smoke.py
```

Kiểm tra API sử dụng model fit trên ảnh synthetic, kiểm tra cả URL `/` và URL HTML
cũ sau khi chuyển frontend vào `app/static/`. Kiểm tra tài liệu đối chiếu liên kết
tương đối. Đối chiếu SHA256 dùng danh sách hiện vật chụp trước khi thay đổi.

## Kết quả cục bộ

| Kiểm tra | Kết quả |
|---|---|
| Ruff lint | Đạt |
| Ruff format | Đạt, 71 file Python |
| Toàn bộ pytest, Python 3.11.15 | 108 passed, 1 skipped |
| Toàn bộ pytest, Python 3.12.13 | 108 passed, 1 skipped |
| Smoke synthetic, Python 3.11 | Đạt, đủ M1–M3 |
| Smoke synthetic, Python 3.12 | Đạt, đủ M1–M3 |
| SHA256 của hiện vật bảo vệ | 568/568 giữ nguyên |
| HTML và PDF sau đổi vị trí | Nội dung byte giữ nguyên |

Test bị skip ở cả hai môi trường là kiểm thử cần `torch_directml`, vì dependency
này không được cài trong hai môi trường CPU dùng cho lượt kiểm tra. Không coi kết
quả này là đã kiểm thử lại backend DirectML. Một deprecation warning từ
Starlette/httpx không làm test thất bại.

Log và JUnit của lượt chạy cục bộ nằm trong `.local/cleanup/`; CI xuất log/JUnit
riêng theo từng phiên bản Python. Số kiểm thử trong bảng phản ánh đúng lượt chạy
cục bộ này, không thay thế trạng thái GitHub Actions của một commit khác.
