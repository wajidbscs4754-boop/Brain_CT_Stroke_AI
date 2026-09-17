from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_loader import BrainCTDataset


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "dataset" / "train_split.csv"
IMAGE_DIR = BASE_DIR / "dataset" / "data"


def test_dataloader():
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = BrainCTDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
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

    # Basic checks
    assert images.shape[0] <= 32
    assert images.shape[1:] == (3, 224, 224)
    assert labels.shape[0] == images.shape[0]

    print("DataLoader test successful")
    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)