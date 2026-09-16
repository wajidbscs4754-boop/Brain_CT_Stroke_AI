import torch
from torchvision import models

# ResNet18 pretrained model
model = models.resnet18(weights="DEFAULT")

# Last layer ko 3 classes ke liye modify karna
model.fc = torch.nn.Linear(
    model.fc.in_features,
    3
)

print("Model created successfully")
print(model.fc)