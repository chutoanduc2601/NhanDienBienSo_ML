import os
import shutil
import random

# Đường dẫn
base_path = 'D:/HocMay/dataset_crnn'
image_dir = os.path.join(base_path, 'images')
label_dir = os.path.join(base_path, 'labels')

train_img_dir = os.path.join(base_path, './train/images')
train_lbl_dir = os.path.join(base_path, './train/labels')

val_img_dir = os.path.join(base_path, './val/images')
val_lbl_dir = os.path.join(base_path, './val/labels')

test_img_dir = os.path.join(base_path, './test/images')
test_lbl_dir = os.path.join(base_path, './test/labels')

# Tạo thư mục nếu chưa có
for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(path, exist_ok=True)

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
random.shuffle(image_files)

# Tính toán số lượng chia
total = len(image_files)
train_count = int(0.7 * total)
val_count = int(0.2 * total)

# Chia tập dữ liệu
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

def move_files(file_list, img_dst, lbl_dst):
    for f in file_list:
        name, _ = os.path.splitext(f)
        img_src_path = os.path.join(image_dir, f)
        lbl_src_path = os.path.join(label_dir, name + '.txt')

        if os.path.exists(lbl_src_path):  # Kiểm tra file label tồn tại
            shutil.copy(img_src_path, os.path.join(img_dst, f))
            shutil.copy(lbl_src_path, os.path.join(lbl_dst, name + '.txt'))

# Di chuyển file
move_files(train_files, train_img_dir, train_lbl_dir)
move_files(val_files, val_img_dir, val_lbl_dir)
move_files(test_files, test_img_dir, test_lbl_dir)

print("Đã chia dataset xong!")
