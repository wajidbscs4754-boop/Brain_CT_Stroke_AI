import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn

from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import BrainCTDataset


# Device
device = torch.device("cpu")


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Test dataset
test_dataset = BrainCTDataset(
    csv_file="dataset/test_split.csv",
    image_dir="dataset/data",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# Model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
       "models/brain_ct_resnet18_weighted.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Class names
class_names = [
    "Normal",
    "Ischemia",
    "Bleeding"
]


# Collect predictions
all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.numpy())


# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# Confusion matrix
print("Confusion Matrix:\n")

matrix = confusion_matrix(
    all_labels,
    all_predictions
)

print(matrix)