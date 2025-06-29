# Tạo val và train list cho CRNN data
import os

def generate_list(split):
    image_dir = f"dataset_crnn/{split}/images"
    label_dir = f"dataset_crnn/{split}/labels"
    output_file = f"data/{split}_list.txt"

    lines = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")

            if not os.path.exists(label_path):
                print(f"⚠️ Warning: Không tìm thấy label cho {filename}")
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label = f.read().strip()

            image_path = os.path.join("..", image_dir, filename)
            lines.append(f"{image_path} {label}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Đã tạo {output_file} với {len(lines)} dòng.")

# Tạo cho train và val
generate_list("train")
generate_list("val")
