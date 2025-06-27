import os
######################
#Kiểm tra nhãn và hình có khớp không
########################################################
sets = ['train', 'val', 'test']
base_dir = 'dataset'

for s in sets:
    image_dir = os.path.join(base_dir, s, 'images')
    label_dir = os.path.join(base_dir, s, 'labels')

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    image_basenames = set(os.path.splitext(f)[0] for f in image_files)
    label_basenames = set(os.path.splitext(f)[0] for f in label_files)

    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames

    if missing_labels:
        print(f"{s}: Thiếu label cho các ảnh: {missing_labels}")
    if missing_images:
        print(f"{s}: Thiếu ảnh cho các label: {missing_images}")

print("Kiểm tra xong. Đảm bảo ảnh và nhãn khớp nhau.")
