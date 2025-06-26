from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CRNNDataset(Dataset):
    def __init__(self, label_file, image_dir, alphabet, imgH=32):
        self.image_dir = image_dir
        self.alphabet = alphabet
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0 is blank
        self.imgH = imgH

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((imgH, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.samples = [line.strip().split(maxsplit=1) for line in lines]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        label_idx = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        input_length = torch.tensor([25], dtype=torch.long)  # tùy theo đầu ra mô hình
        target_length = torch.tensor([len(label)], dtype=torch.long)

        return image, label_tensor, input_length, target_length
