import pandas as pd

# Training CSV load karein
df = pd.read_csv("dataset/train_split.csv")

# CSV ki third column mein numeric labels hain
label_column = df.columns[2]

# Class labels ka count
counts = df[label_column].value_counts().sort_index()

class_names = {
    0: "Normal",
    1: "Ischemia",
    2: "Bleeding"
}

print("Training class distribution:\n")

for label, count in counts.items():
    print(f"{class_names[int(label)]}: {count} images")