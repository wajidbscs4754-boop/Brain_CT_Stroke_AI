import pandas as pd

# Original training CSV load karein
df = pd.read_csv("dataset/train_split.csv")

# CSV columns
path_column = df.columns[0]
class_column = df.columns[1]
label_column = df.columns[2]

# Har class ko alag karein
normal = df[df[label_column] == 0]
ischemia = df[df[label_column] == 1]
bleeding = df[df[label_column] == 2]

# Normal ki sirf 1000 images randomly select karein
normal_sample = normal.sample(
    n=800,
    random_state=42
)

# Balanced dataset combine karein
balanced_df = pd.concat(
    [normal_sample, ischemia, bleeding],
    ignore_index=True
)

# Rows ko shuffle karein
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# New CSV save karein
balanced_df.to_csv(
    "dataset/balanced_train_split.csv",
    index=False
)

print("Balanced CSV created successfully")
print("\nClass distribution:")

print(
    balanced_df[label_column]
    .value_counts()
    .sort_index()
)