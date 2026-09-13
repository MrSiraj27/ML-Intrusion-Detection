from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

for name, path in [("Training", train_path), ("Testing", test_path)]:
    df = pd.read_csv(path)
    print(f"\n=== {name} set: {path.name} ===")
    print("Row count:", len(df))
    print("Column count:", len(df.columns))
    print("Column names:", list(df.columns))
    print("\nlabel value counts:")
    print(df["label"].value_counts())
    print("\nattack_cat value counts:")
    print(df["attack_cat"].value_counts())
