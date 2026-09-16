from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BrainCTDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.image_dir / row.iloc[0]

        label = int(row.iloc[2])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Dataset create
train_dataset = BrainCTDataset(
    csv_file="dataset/train_split.csv",
    image_dir="dataset/data",
    transform=transform
)


# Test dataset
image, label = train_dataset[0]

print("Dataset loaded successfully")
print("Total images:", len(train_dataset))
print("Image shape:", image.shape)
print("Label:", label)