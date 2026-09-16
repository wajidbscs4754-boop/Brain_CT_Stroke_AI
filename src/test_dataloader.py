import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import BrainCTDataset


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Dataset
train_dataset = BrainCTDataset(
    csv_file="dataset/train_split.csv",
    image_dir="dataset/data",
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# Get first batch
images, labels = next(iter(train_loader))


print("DataLoader test successful")
print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Labels:", labels)