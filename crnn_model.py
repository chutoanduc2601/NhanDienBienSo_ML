# crnn_model.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.crnn import CRNN  # đường dẫn đúng tới file crnn.py
from utils.strLabelConverter import strLabelConverter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'crnn.pytorch'))

from models.crnn import CRNN
from utils.strLabelConverter import strLabelConverter


class CRNNRecognizer:
    def __init__(self, model_path='crnn.pth', imgH=32, nc=1, nclass=37, nh=256):
        self.model = CRNN(imgH, nc, nclass, nh)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.converter = strLabelConverter('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.')

        self.transform = transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, img_np):
        image = Image.fromarray(img_np)
        image = self.transform(image).unsqueeze(0)  # Add batch dim

        preds = self.model(image)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)

        pred_text = self.converter.decode(preds_index.data, preds_size=torch.IntTensor([preds.size(0)]))
        return pred_text
