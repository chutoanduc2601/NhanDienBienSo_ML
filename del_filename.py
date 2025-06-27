import os

# File chứa nhãn gốc (6 cột)
location_file = 'dataset/location.txt'

# Thư mục lưu nhãn YOLO chuẩn
label_output_dir = './dataset/labels'
os.makedirs(label_output_dir, exist_ok=True)

# Đọc từng dòng và xử lý
with open(location_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) != 6:
        continue  # bỏ qua dòng không hợp lệ

    filename, class_id, x_center, y_center, width, height = parts

    # Tạo tên file nhãn .txt tương ứng
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(label_output_dir, label_filename)

    # Ghi đúng định dạng YOLO: class_id x_center y_center width height
    with open(label_path, 'w') as label_file:
        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Đã tạo lại file nhãn đúng định dạng YOLO (5 cột)!")
