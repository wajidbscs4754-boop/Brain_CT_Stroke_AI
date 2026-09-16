import pandas as pd
from pathlib import Path

DATASET_PATH = Path("dataset")

csv_files = [
    "train_split.csv",
    "val_split.csv",
    "test_split.csv"
]

print("\n========== CSV DATASET CHECK ==========\n")

for csv_name in csv_files:

    csv_path = DATASET_PATH / csv_name

    print(f"\n📄 File: {csv_name}")

    df = pd.read_csv(csv_path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\n" + "-" * 40)

print("\nCSV check completed.")