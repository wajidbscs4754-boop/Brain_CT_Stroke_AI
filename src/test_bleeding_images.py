from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "brain_ct_resnet18_weighted.pth"
IMAGE_DIR = BASE_DIR / "dataset" / "data" / "Bleeding"


# =========================================================
# Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# Classes
# =========================================================

CLASS_NAMES = [
    "Normal",
    "Ischemia",
    "Bleeding"
]


def test_bleeding_images():

    # =====================================================
    # Check paths
    # =====================================================

    assert MODEL_PATH.exists(), "Trained model not found"

    assert IMAGE_DIR.exists(), "Bleeding image folder not found"

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
    # Image preprocessing
    # =====================================================

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # Get images
    # =====================================================

    image_files = sorted(
        [
            file
            for file in IMAGE_DIR.iterdir()
            if file.is_file()
            and file.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )[:5]

    assert len(image_files) > 0, "No images found in Bleeding folder"

    # =====================================================
    # Test predictions
    # =====================================================

    for image_path in image_files:

        image = Image.open(image_path).convert("RGB")

        input_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

        # Prediction must be one of 3 classes
        assert 0 <= predicted_index < 3

        prediction = CLASS_NAMES[predicted_index]

        assert prediction in CLASS_NAMES

        print(
            f"{image_path.name} -> {prediction}"
        )

    print("Bleeding images test successful")