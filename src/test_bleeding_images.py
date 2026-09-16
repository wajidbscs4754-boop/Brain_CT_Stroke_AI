from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "brain_ct_resnet18_weighted.pth"

IMAGE_DIR = BASE_DIR / "dataset" / "data" / "Bleeding"


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Ischemia",
    2: "Bleeding",
}


# =========================================================
# CHECK FILES
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")


# =========================================================
# LOAD MODEL
# =========================================================

device = torch.device("cpu")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


# =========================================================
# IMAGE PREPROCESSING
# Must match prediction/web app preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# SELECT 5 IMAGES
# =========================================================

image_files = sorted(
    [
        file
        for file in IMAGE_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]
)[:5]


if len(image_files) == 0:
    raise ValueError("No images found in Bleeding folder.")


# =========================================================
# TEST IMAGES
# =========================================================

print("\n==============================================================")
print("              BLEEDING IMAGES TEST")
print("==============================================================")

correct_predictions = 0

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

    prediction = CLASS_NAMES[predicted_index]

    normal = probabilities[0][0].item() * 100
    ischemia = probabilities[0][1].item() * 100
    bleeding = probabilities[0][2].item() * 100

    if prediction == "Bleeding":
        correct_predictions += 1

    print(f"\nImage: {image_path.name}")
    print(f"Normal:    {normal:.2f}%")
    print(f"Ischemia:  {ischemia:.2f}%")
    print(f"Bleeding:  {bleeding:.2f}%")
    print(f"Prediction: {prediction}")


# =========================================================
# FINAL SUMMARY
# =========================================================

total_images = len(image_files)

accuracy = (
    correct_predictions / total_images
) * 100

print("\n==============================================================")
print("                    FINAL SUMMARY")
print("==============================================================")

print(f"Total Bleeding images tested: {total_images}")
print(f"Correctly predicted as Bleeding: {correct_predictions}")
print(f"Bleeding test accuracy: {accuracy:.2f}%")