from pathlib import Path

# Dataset ka path
DATASET_PATH = Path("dataset/data")

# Hamari 3 classes
classes = ["Bleeding", "Ischemia", "Normal"]

# Image extensions
image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

print("\n========== DATASET ANALYSIS ==========\n")

total_images = 0

for class_name in classes:

    folder = DATASET_PATH / class_name

    images = [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    count = len(images)
    total_images += count

    print(f"{class_name}: {count} images")

print("\n--------------------------------------")
print(f"Total images: {total_images}")
print("--------------------------------------")

print("\nAnalysis completed.")