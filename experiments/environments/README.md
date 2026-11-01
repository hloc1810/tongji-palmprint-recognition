# Môi trường thực nghiệm

- classical.txt: dependency snapshot của benchmark M1–M3, Python 3.11.15 trên Windows CPU.
- cnn-directml.txt: dependency snapshot của ResNet18 ba seed, Python 3.11.15, torch 2.4.1/torchvision 0.19.1 và DirectML 0.2.5.dev240914 trên Radeon RX 5500M.

Đã bỏ đường dẫn editable-install của máy khỏi các snapshot. Không coi đây là lockfile đa nền tảng. Các phiên bản tương thích cài đặt chung được khai báo trong pyproject.toml; backend DirectML cần môi trường riêng. Kết quả embedding do người thực hiện chạy với môi trường Python 3.12 cục bộ; không gán snapshot DirectML này cho lượt embedding.
