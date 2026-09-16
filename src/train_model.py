import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn, optim

from dataset_loader import BrainCTDataset


# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# 2. Image transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -----------------------------
# 3. Load datasets
# -----------------------------
train_dataset = BrainCTDataset(
    csv_file="dataset/balanced_train_split.csv",
    image_dir="dataset/data",
    transform=transform
)

val_dataset = BrainCTDataset(
    csv_file="dataset/val_split.csv",
    image_dir="dataset/data",
    transform=transform
)


# -----------------------------
# 4. Create DataLoaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# 5. Load ResNet18
# -----------------------------
model = models.resnet18(weights="DEFAULT")

# Final layer for 3 classes
model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model = model.to(device)


# -----------------------------
# 6. Loss and optimizer
# -----------------------------
# Class weights
# Order: Normal, Ischemia, Bleeding
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# 7. Training settings
# -----------------------------
epochs = 5


# -----------------------------
# 8. Training loop
# -----------------------------
for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# -----------------------------
# 9. Save trained model
# -----------------------------
model_path = "models/brain_ct_resnet18_weighted.pth"

torch.save(model.state_dict(), model_path)

print("Training completed successfully")
print("Model saved at:", model_path)
