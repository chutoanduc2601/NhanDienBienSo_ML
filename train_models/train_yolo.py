from ultralytics import YOLO

model = YOLO('../models/yolov8s.pt')

model.train(
    data='license_plate_data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='lp_yolov8_training',
    workers=4
)
#Train 8s khong phai 8n , 8s > 8n luu y
