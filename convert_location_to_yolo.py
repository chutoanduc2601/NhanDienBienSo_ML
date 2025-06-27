import os

# Đường dẫn file location.txt
location_file = 'dataset/location.txt'

# Thư mục lưu nhãn sau khi tách
label_output_dir = 'dataset/labels'
os.makedirs(label_output_dir, exist_ok=True)

# Đọc file location.txt
with open(location_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 6:
        continue  # Bỏ qua dòng không hợp lệ

    filename, class_id, x_center, y_center, width, height = parts

    # Tạo tên file .txt tương ứng với ảnh
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(label_output_dir, label_filename)

    # Ghi đúng định dạng gốc vào file nhãn
    with open(label_path, 'w') as label_file:
        label_file.write(f"{filename} {class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Đã chia location.txt thành từng file nhãn riêng!")
