import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from crnn_model import CRNN
from crnn_dataset import CRNNDataset

# Cấu hình
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
num_classes = len(alphabet) + 1  # +1 cho ký tự blank
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
train_dataset = CRNNDataset("../dataset_yolov8/images_test/labels.txt", "dataset_yolov8/images_test", alphabet)


def my_collate_fn(batch):
    images, labels, input_lengths, target_lengths = zip(*batch)
    images = torch.stack(images, 0)

    # Ghép tất cả nhãn thành một tensor dài, kiểu CTC cần
    targets = torch.cat(labels)

    input_lengths = torch.tensor([25] * len(batch), dtype=torch.long)  # hoặc tính đúng theo output CRNN
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    return images, targets, input_lengths, target_lengths
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,collate_fn=my_collate_fn)

# Mô hình
model = CRNN(imgH=32, nc=1, nclass=num_classes, nh=256).to(device)
criterion = nn.CTCLoss(blank=0, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for images, targets, input_lengths, target_lengths in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)  # [W, B, C]
        preds = preds.log_softmax(2)
        loss = criterion(preds, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")




# Lưu mô hình
torch.save(model, "crnn_model.pth")
