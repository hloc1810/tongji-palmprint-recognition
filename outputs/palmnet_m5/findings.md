# PCA-Gabor PalmNet — kết quả đã kiểm tra

6.000 gallery Session 1 / 6.000 probe Session 2, 600 Palm ID.

| Phương pháp | Số chiều | Rank-1 (%) | Rank-5 (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| M5 PalmNet + 1-NN (full) | 12,288,000 | 99.98 | 100.00 | 99.98 |
| PalmNet + Variance-1024 + 1-NN | 1,024 | 98.78 | 99.32 | 98.61 |
| PalmNet + Variance-1024 + Linear SVM | 1,024 | 98.85 | 99.45 | 98.69 |
| PalmNet + Variance-1024 + CRC_RLS | 1,024 | 98.73 | 99.33 | 98.58 |

Các nhánh 1.024 chiều dùng cùng phép chọn phương sai fit trên Session 1.
So sánh SVM/CRC với 1-NN 1.024 chiều để tách tác động classifier.
Đây là cấu hình mở rộng cố định, chưa tuning bằng validation Session 1.

Nguồn PalmNet: f0a12e6d5f585c7eb0bf5c6804a98bd1d6587e22
Manifest SHA-256: eaeb9ac0a5e0b55050b105031d70aa895c609a4d3fcb018b504c1f0bc55055ea
Model SHA-256: a2362e89ed7a0ca829df0c90d45328367fc48df6ec26ce9c106f6bdcf4ff5b05

Timing là theo batch và gồm IO cache, không phải single-query latency.
