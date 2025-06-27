import cv2
import os

# Cấu hình
image_path = "0000_00532_b_jpg.rf.4ee48829fc2d3d1172462159643a2e6d.jpg"  # đường dẫn ảnh
label_path = "0000_00532_b_jpg.rf.4ee48829fc2d3d1172462159643a2e6d.txt"  # file nhãn YOLO tương ứng

# Đọc ảnh
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Đọc file nhãn và vẽ box
with open(label_path, "r") as f:
    for line in f:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert từ YOLO format sang pixel
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Vẽ khung
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"GT class {int(class_id)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Hiển thị
cv2.imshow("Ground Truth Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
