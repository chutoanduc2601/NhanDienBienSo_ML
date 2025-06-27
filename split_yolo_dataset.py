import os
import shutil
import random
from pathlib import Path

# === Cấu hình đường dẫn gốc ===
root_dir = Path("D:/HocMay/dataset")
image_dir = root_dir / "image"
label_dir = root_dir / "labels"

# === Tạo thư mục đầu ra ===
splits = ['train', 'val', 'test']
ratios = [0.7, 0.2, 0.1]
for split in splits:
    (root_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (root_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# === Ghép file ảnh và nhãn tương ứng ===
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
paired = []

for img_file in image_files:
    # Loại bỏ cả đuôi .jpg và phần mở rộng .rf...
    name_stem = '.'.join(img_file.split('.')[:-1])  # hoặc Path(img_file).stem nếu dùng pathlib
    label_file = name_stem + ".txt"
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        paired.append((img_file, label_file))
    else:
        print(f"Không tìm thấy nhãn tương ứng cho {img_file}")

# === Chia dữ liệu theo tỉ lệ ===
random.shuffle(paired)
n = len(paired)
n_train = int(n * ratios[0])
n_val = int(n * ratios[1])

split_data = {
    'train': paired[:n_train],
    'val': paired[n_train:n_train + n_val],
    'test': paired[n_train + n_val:]
}

# === Sao chép ảnh và nhãn ===
for split, items in split_data.items():
    for img_name, label_name in items:
        shutil.copy(image_dir / img_name, root_dir / split / "images" / img_name)
        shutil.copy(label_dir / label_name, root_dir / split / "labels" / label_name)

print("✅ Hoàn tất chia dataset thành train/val/test.")
