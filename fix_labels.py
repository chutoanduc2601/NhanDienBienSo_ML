import os
################################
# Cho label id về 0 trước khi train
#############################################
label_root = './dataset/labels'  # Chỉnh đúng nếu nhãn gốc nằm ở đây

for split in ['train', 'val', 'test']:
    label_dir = os.path.join('dataset', split, 'labels')
    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            path = os.path.join(label_dir, fname)
            with open(path, 'r') as f:
                lines = f.readlines()
            with open(path, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = '0'  # Gán class_id = 0
                        f.write(' '.join(parts) + '\n')
cache_file = "dataset/train/labels.cache"
if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"🗑️ Đã xoá {cache_file}")
print("✅ Đã sửa tất cả nhãn về class_id = 0.")
