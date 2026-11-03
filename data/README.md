# Tongji Contactless Palmprint Dataset

Nguồn dữ liệu: [trang tác giả Tongji](https://cslinzhang.github.io/ContactlessPalm/).
Tải hai bộ **Original Images** và **ROI Images** từ liên kết trên trang này theo
điều kiện sử dụng của đơn vị cung cấp. Không đưa ảnh dataset vào Git.

## Bố trí trên máy

```text
data/
├── raw/
│   ├── session1/00001.bmp … 06000.bmp       # Official ROI 128×128
│   └── session2/00001.bmp … 06000.bmp
├── Original Images/
│   ├── session1/00001.tiff … 06000.tiff     # Ảnh bàn tay toàn phần
│   └── session2/00001.tiff … 06000.tiff
├── metadata/                               # manifest, local_dataset, audit
└── splits/                                 # fold Session 1
```

Mỗi bộ có 12.000 ảnh, 600 Palm ID, hai session, 10 ảnh/ID/session. Đuôi ảnh có
thể phụ thuộc bản phân phối; decoder chấp nhận các định dạng được khai báo
trong `src/palmprint/data/dataset.py`. Không đổi tên stem số của ảnh.

## Nhãn và ghép cặp

`00001`–`00010` là Palm ID 1, `00011`–`00020` là ID 2, v.v.
Công thức: `palm_id = (int(stem) - 1) // 10 + 1`. Hai tay là hai Palm ID khác nhau.
UID luôn có session, ví dụ `session1/00001` và `session2/00001`.

Original và Official ROI được ghép theo **cùng session và cùng stem**.
Không đưa ảnh Original vào `raw/`; mã benchmark kiểm tra 128×128 và từ chối
ảnh toàn bàn tay. Custom ROI dùng extractor riêng, không resize toàn ảnh.

## Manifest và đường dẫn cục bộ

`metadata/` và `splits/` trong repository thuộc lần chạy đã công bố, có thể chứa
đường dẫn máy cũ. Không đổi chúng trong workspace đã khóa. Khi chạy trên máy
mới, tạo workspace riêng và gọi `palmprint --workspace <workspace> prepare`;
hướng dẫn và chuỗi lệnh nằm trong [tài liệu tái lập](../docs/REPRODUCIBILITY.md).

Pilot ROI mặc định đọc trực tiếp `data/raw` và `data/Original Images` của repository
hiện tại, có thể ghi đè đường dẫn bằng các flag; không phụ thuộc đường dẫn cũ
trong metadata benchmark. Dataset, model và cache cục bộ không được xóa như
file tạm khi dọn repository.
