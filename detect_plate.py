from ultralytics import YOLO
import cv2

# Load model đã huấn luyện
model = YOLO(r'D:\nhandienphuongtienvabiensoxe\runs\detect\lp_yolov8_training3\weights\best.pt')

# Ảnh test
img_path = r'D:\HocMay\dataset\image\0000_00532_b.jpg'
results = model(img_path, conf=0.25)

# Hiển thị kết quả
results[0].show()  # Hiển thị ảnh có bounding box

# Hoặc lưu kết quả
results[0].save(filename='output_detected.jpg')
