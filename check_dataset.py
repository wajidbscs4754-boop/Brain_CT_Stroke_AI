from pathlib import Path

# Dataset ka path
DATASET_PATH = Path("dataset/data")

# Hamari 3 classes
classes = ["Bleeding", "Ischemia", "Normal"]

# Supported image formats
image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

print("\n========== DATASET CHECK ==========\n")

total_images = 0

for class_name in classes:

    folder = DATASET_PATH / class_name

    if not folder.exists():
        print(f"❌ {class_name}: Folder not found")
        continue

    images = [
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    count = len(images)
    total_images += count

    print(f"✅ {class_name}: {count} images")

print("\n-----------------------------------")
print(f"Total images: {total_images}")
print("-----------------------------------")

# CSV files check
print("\nCSV Files:\n")

csv_files = [
    "train_split.csv",
    "val_split.csv",
    "test_split.csv"
]

for csv_name in csv_files:

    csv_path = Path("dataset") / csv_name

    if csv_path.exists():
        print(f"✅ {csv_name}: Found")
    else:
        print(f"❌ {csv_name}: Not found")

print("\nDataset check completed.")