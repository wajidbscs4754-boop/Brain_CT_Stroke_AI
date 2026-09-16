import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset_loader import BrainCTDataset


# Device
device = torch.device("cpu")


# Image transformation
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


# Load pretrained ResNet18
model = models.resnet18(weights="DEFAULT")

# Change final layer for 3 classes
model.fc = torch.nn.Linear(
    model.fc.in_features,
    3
)

model = model.to(device)
model.eval()


# Get one batch
images, labels = next(iter(train_loader))

images = images.to(device)

# Model prediction
with torch.no_grad():
    outputs = model(images)

predictions = torch.argmax(outputs, dim=1)


print("Model prediction test successful")
print("Input images shape:", images.shape)
print("Model output shape:", outputs.shape)
print("Predictions shape:", predictions.shape)
print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", labels[:10])