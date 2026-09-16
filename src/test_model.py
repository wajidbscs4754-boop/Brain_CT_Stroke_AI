import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report

from dataset_loader import BrainCTDataset


# =========================================================
# 1. Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# 2. Image transformation
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================================================
# 3. Test dataset
# =========================================================

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


# =========================================================
# 4. Create model
# =========================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)


# =========================================================
# 5. Load trained model
# =========================================================

model.load_state_dict(
    torch.load(
        "models/brain_ct_resnet18_weighted.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# 6. Class names
# =========================================================

class_names = [
    "Normal",
    "Ischemia",
    "Bleeding"
]


# =========================================================
# 7. Store predictions
# =========================================================

all_labels = []
all_predictions = []


# =========================================================
# 8. Evaluate model
# =========================================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =========================================================
# 9. Overall accuracy
# =========================================================

correct = sum(
    prediction == label
    for prediction, label
    in zip(all_predictions, all_labels)
)

total = len(all_labels)

test_accuracy = (
    100 * correct / total
)


# =========================================================
# 10. Confusion matrix
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2]
)


# =========================================================
# 11. Print results
# =========================================================

print("\n========================================")
print("          TEST SET EVALUATION")
print("========================================")

print(f"Total test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Overall Test Accuracy: {test_accuracy:.2f}%")


# =========================================================
# 12. Class-wise results
# =========================================================

print("\n========================================")
print("          CLASS-WISE RESULTS")
print("========================================")

for index, class_name in enumerate(class_names):

    total_class = cm[index].sum()
    correct_class = cm[index][index]

    class_accuracy = (
        100 * correct_class / total_class
        if total_class > 0
        else 0
    )

    print(f"\n{class_name}")
    print(f"Total images: {total_class}")
    print(f"Correct: {correct_class}")
    print(f"Incorrect: {total_class - correct_class}")
    print(f"Accuracy: {class_accuracy:.2f}%")


# =========================================================
# 13. Confusion matrix
# =========================================================

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print(
    "                  Predicted"
)

print(
    "             Normal  Ischemia  Bleeding"
)

print(
    f"Actual Normal    {cm[0][0]:4d}"
    f"    {cm[0][1]:4d}"
    f"    {cm[0][2]:4d}"
)

print(
    f"Actual Ischemia  {cm[1][0]:4d}"
    f"    {cm[1][1]:4d}"
    f"    {cm[1][2]:4d}"
)

print(
    f"Actual Bleeding  {cm[2][0]:4d}"
    f"    {cm[2][1]:4d}"
    f"    {cm[2][2]:4d}"
)


# =========================================================
# 14. Classification report
# =========================================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        labels=[0, 1, 2],
        zero_division=0
    )
)


# =========================================================
# 15. Final
# =========================================================

print("========================================")
print("Test evaluation completed.")
print("========================================")