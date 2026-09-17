from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn

from dataset_loader import BrainCTDataset


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "dataset" / "train_split.csv"
IMAGE_DIR = BASE_DIR / "dataset" / "data"
MODEL_PATH = BASE_DIR / "models" / "brain_ct_resnet18_weighted.pth"


# =========================================================
# Device
# =========================================================

device = torch.device("cpu")


def test_model_prediction():

    # =====================================================
    # Image transformation
    # =====================================================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # =====================================================
    # Dataset
    # =====================================================

    train_dataset = BrainCTDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # =====================================================
    # Create model
    # =====================================================

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        3
    )

    # =====================================================
    # Load trained model
    # =====================================================

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    # =====================================================
    # Get one batch
    # =====================================================

    images, labels = next(iter(train_loader))

    images = images.to(device)

    # =====================================================
    # Prediction
    # =====================================================

    with torch.no_grad():
        outputs = model(images)

    predictions = torch.argmax(outputs, dim=1)

    # =====================================================
    # Checks
    # =====================================================

    assert outputs.shape[0] == images.shape[0]
    assert outputs.shape[1] == 3
    assert predictions.shape[0] == labels.shape[0]

    assert torch.all(predictions >= 0)
    assert torch.all(predictions < 3)

    print("Model prediction test successful")
    print("Input images shape:", images.shape)
    print("Model output shape:", outputs.shape)
    print("Predictions shape:", predictions.shape)
    print("First 10 predictions:", predictions[:10])
    print("First 10 actual labels:", labels[:10])