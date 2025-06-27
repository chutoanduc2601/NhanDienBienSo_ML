import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# === Cấu hình đường dẫn ===
location_file = 'dataset/location.txt'
image_dir = 'dataset/image'
base_dir = 'dataset'
splits = ['train', 'val', 'test']

# Tạo thư mục
for split in splits:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Đọc file
with open(location_file, 'r') as f:
    lines = [line.strip() for line in f if len(line.strip().split()) == 6]

# Chia tập
train_lines, temp_lines = train_test_split(lines, test_size=0.3, random_state=42)
val_lines, test_lines = train_test_split(temp_lines, test_size=1/3, random_state=42)

split_data = {
    'train': train_lines,
    'val': val_lines,
    'test': test_lines
}

for split, lines in split_data.items():
    print(f"🔹 Đang xử lý {split} ({len(lines)} ảnh)...")

    for line in lines:
        parts = line.split()
        filename, class_id, x, y, w, h = parts
        x, y, w, h = map(float, [x, y, w, h])

        # Đường dẫn ảnh
        src_img_path = os.path.join(image_dir, filename)
        dst_img_path = os.path.join(base_dir, split, 'images', filename)

        if not os.path.exists(src_img_path):
            print(f"⚠️ Không tìm thấy ảnh: {filename}")
            continue

        # Đọc kích thước ảnh
        img = cv2.imread(src_img_path)
        img_h, img_w = img.shape[:2]

        # ✅ Normalize theo chuẩn YOLO
        x_center_norm = x / img_w
        y_center_norm = y / img_h
        width_norm = w / img_w
        height_norm = h / img_h

        # Kiểm tra hợp lệ
        if not all(0 <= v <= 1 for v in [x_center_norm, y_center_norm, width_norm, height_norm]):
            print(f"⚠️ Nhãn lỗi (out of bounds): {filename} [{x}, {y}, {w}, {h}]")
            continue

        # Copy ảnh
        shutil.copy2(src_img_path, dst_img_path)

        # Ghi nhãn chuẩn
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(base_dir, split, 'labels', label_filename)

        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

print("✅ Đã chia ảnh + nhãn và chuẩn hóa đúng định dạng YOLO!")
