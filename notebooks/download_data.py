import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
print("Downloaded to:", path)

src_dir = Path(path)
for name in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
    matches = list(src_dir.rglob(name))
    if not matches:
        print(f"WARNING: {name} not found under {src_dir}")
        continue
    dest = DATA_DIR / name
    shutil.copy(matches[0], dest)
    print(f"Copied {name} -> {dest}")
