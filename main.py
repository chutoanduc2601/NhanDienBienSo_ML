import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import os
# from crnn_model import CRNNRecognizer

# Tạo thư mục nếu chưa có
os.makedirs('images_result', exist_ok=True)

# Khởi tạo mô hình
coco_model = YOLO('models/yolov8s.pt')  # Phát hiện phương tiện
# plate_model = YOLO('license_plate_detector.pt')  # Phát hiện biển số
plate_model = YOLO('./models/best.pt')  # Sử dụng model đã huấn luyện
ocr_reader = easyocr.Reader(['en'])  # OCR tiếng Anh

# Đọc ảnh
img = cv2.imread('dataset_yolov8/images_test/image11.jpg')
vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Phát hiện phương tiện
coco_detections = coco_model(img)[0]
for det in coco_detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = det
    if int(class_id) in vehicles:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Phát hiện biển số
plate_detections = plate_model(img)[0]
for i, det in enumerate(plate_detections.boxes.data.tolist()):
    x1, y1, x2, y2, score, class_id = det

    # Cắt ảnh biển số
    plate_crop = img[int(y1):int(y2), int(x1):int(x2)]

    # # Chuyển sang ảnh xám
    # gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    #
    # # Cân bằng sáng với CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)
    #
    # # Làm mịn để giảm nhiễu nhưng giữ cạnh ký tự
    # blur = cv2.bilateralFilter(gray, 11, 17, 17)
    #
    # # Dùng Otsu threshold để chọn ngưỡng phù hợp
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # # Loại bỏ các vùng nhiễu nhỏ
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #
    # # Phóng to để OCR dễ đọc hơn
    # thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    #
    # #############################################################
    # # OCR từng nửa ảnh (2 dòng)
    # h, w = thresh.shape
    # upper_half = thresh[0:int(h / 2), :]
    # lower_half = thresh[int(h / 2):, :]
    #
    # upper_text = ocr_reader.readtext(upper_half, detail=0)
    # lower_text = ocr_reader.readtext(lower_half, detail=0)
    #
    # # Ghép kết quả từng nửa
    # plate_text = ''
    # if upper_text:
    #     plate_text += ' '.join([t.strip() for t in upper_text])
    # if lower_text:
    #     plate_text += ' ' + ' '.join([t.strip() for t in lower_text])
    #
    # # Nếu quá ngắn (chỉ có 1 dòng, không đầy đủ), thử lại toàn biển
    # if len(plate_text.replace(" ", "")) < 6:
    #     fallback_text = ocr_reader.readtext(thresh, detail=0)
    #     if fallback_text:
    #         plate_text = ' '.join([t.strip() for t in fallback_text])
    def get_best_rotation_ocr(image, ocr_reader):
        angles = [0, 90, 180, 270]
        best_text = ''
        best_thresh = None

        for angle in angles:
            rotated = image.copy()
            if angle != 0:
                rot_code = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE
                }[angle]
                rotated = cv2.rotate(rotated, rot_code)

            # Chuyển xám và CLAHE để làm rõ chi tiết
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Làm mịn và phân ngưỡng
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Khử nhiễu
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            texts = ocr_reader.readtext(thresh, detail=0)
            clean = ' '.join(texts).upper().replace("\n", " ").replace("–", "-").replace("_", "-").replace("|",
                                                                                                           "1").strip()

            if len(clean.replace(" ", "")) > len(best_text.replace(" ", "")):
                best_text = clean
                best_thresh = thresh

        return best_text, best_thresh




    # Ghép và làm sạch văn bản
    # Thử xoay và OCR để chọn kết quả tốt nhất
    plate_text, thresh = get_best_rotation_ocr(plate_crop, ocr_reader)

    plate_text = plate_text.replace('\n', ' ').replace("–", "-").replace("_", "-").replace("|", "1").replace('  ', ' ').strip()

    # Sửa các lỗi OCR thường gặp
    replacements = {
        "O": "0", "I": "1", "|": "1", "L": "1",
        "Z": "2", "A": "4", "B": "8", "G": "6", "S": "5", "T": "7",
        "5T": "57", "S7": "57", "8O": "80", "B0": "80",
    }
    for wrong, right in replacements.items():
        plate_text = plate_text.replace(wrong, right)

    # Format lại: chuẩn hóa thành dạng XX-XXX.XX
    # Làm sạch ký tự
    raw_plate = plate_text.replace(" ", "").replace("-", "").replace(".", "")

    # Định dạng lại theo dạng XX-XX XXX.XX
    if len(raw_plate) >= 9:
        # VD: 59F168955 -> 59-F1 689.55
        plate_text = f"{raw_plate[:2]}-{raw_plate[2:4]} {raw_plate[4:7]}.{raw_plate[7:9]}"
    else:
        # fallback nếu không đủ ký tự
        plate_text = raw_plate

    # Làm sạch văn bản: bỏ ký tự thừa
    plate_text = plate_text.replace('\n', ' ').replace('  ', ' ').strip()

    # Làm sạch văn bản: bỏ ký tự không hợp lệ, chuẩn hóa cách viết
    plate_text = plate_text.replace('\n', ' ').replace('  ', ' ').strip()

    #####################################################
    # Vẽ lên ảnh gốc
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(img, plate_text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị và lưu ảnh threshold
    cv2.imshow(f'Thresh Plate {i}', thresh)
    cv2.imwrite(f'images_result/thresh_plate_{i}.png', thresh)

# Lưu ảnh kết quả
cv2.imwrite('images_result/result_detected.png', img)
# Mã tỉnh và tên tương ứng
province_codes = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng",
    "16": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ",
    "20": "Thái Nguyên", "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
    "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La", "27": "Điện Biên",
    "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội",
    "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình", "36": "Thanh Hóa",
    "37": "Nghệ An", "38": "Hà Tĩnh", "43": "Đà Nẵng", "47": "Đắk Lắk",
    "48": "Đắk Nông", "49": "Lâm Đồng", "50": "TP.HCM", "51": "TP.HCM",
    "52": "TP.HCM", "53": "TP.HCM", "54": "TP.HCM", "55": "TP.HCM", "56": "TP.HCM",
    "57": "TP.HCM", "58": "TP.HCM", "59": "TP.HCM", "60": "Đồng Nai", "61": "Bình Dương",
    "62": "Long An", "63": "Tiền Giang", "64": "Vĩnh Long", "65": "Cần Thơ",
    "66": "Đồng Tháp", "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau",
    "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình",
    "74": "Quảng Trị", "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định",
    "78": "Phú Yên", "79": "Khánh Hòa", "80": "Cơ quan TW", "81": "Gia Lai",
    "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh", "85": "Ninh Thuận",
    "86": "Bình Thuận", "88": "Vĩnh Phúc", "89": "Hưng Yên", "90": "Hà Nam",
    "92": "Quảng Nam", "93": "Bình Phước", "94": "Bạc Liêu", "95": "Hậu Giang",
    "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh"
}

# Làm sạch văn bản: bỏ ksi tự khong hop le
plate_text = plate_text.replace('\n', ' ').replace('  ', ' ').strip()
if len(plate_text.replace(" ", "")) < 7:
    full_ocr = ocr_reader.readtext(plate_crop, detail=0)
    if full_ocr:
        alt_text = ' '.join(full_ocr).upper()
        if len(alt_text.replace(" ", "")) > len(plate_text.replace(" ", "")):
            plate_text = alt_text

# Tách 2 chữ số đầu để tra tỉnh
plate_code = plate_text.strip().replace(" ", "").replace("-", "")
province_name = "Không xác định"
if len(plate_code) >= 2:
    code_prefix = plate_code[:2]
    province_name = province_codes.get(code_prefix, "Khong ro")

# In ra kết quả
print(f"Biển số: {plate_text} - Tỉnh/TP: {province_name}")

# Hiển thị tên tỉnh lên ảnh
cv2.putText(img, province_name, (int(x1), int(y2) + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

# Hiển thị ảnh kết quả
cv2.imshow('Detected Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
