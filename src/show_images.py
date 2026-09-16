from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = Path("dataset/data")

classes = ["Bleeding", "Ischemia", "Normal"]

for class_name in classes:

    folder = dataset_path / class_name

    images = list(folder.glob("*"))

    if len(images) == 0:
        print(f"No images found in {class_name}")
        continue

    image_path = images[0]

    image = Image.open(image_path)

    plt.imshow(image, cmap="gray")
    plt.title(class_name)
    plt.axis("off")
    plt.show()
    