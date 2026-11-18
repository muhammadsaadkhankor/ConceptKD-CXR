"""
Load a small subset of the CheXpert dataset.

- Reads train.csv and valid.csv from ../Dataset
- Keeps only these 5 labels:
    Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
- For EACH label, picks IMAGES_PER_LABEL images where that label is
  positive (1.0). Images may have other positive findings too
  (union findings are allowed).
- Resolves CSV paths (CheXpert-v1.0-small/...) to ../Dataset/...
- Writes the subset to chexpert_subset.csv next to this script.

Run with:  python data.py
"""

import os
import pandas as pd

# ---------------- Config ----------------
DATASET_DIR = os.path.join("..", "Dataset")          # ../Dataset
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")
OUTPUT_CSV = "chexpert_subset.csv"

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

IMAGES_PER_LABEL = 20   # images per label -> up to 100 total (with overlap)
RANDOM_SEED = 42
# -----------------------------------------


def resolve_path(csv_path: str) -> str:
    """Map 'CheXpert-v1.0-small/train/...' -> '../Dataset/train/...'"""
    rel = csv_path.replace("CheXpert-v1.0-small/", "", 1)
    return os.path.join(DATASET_DIR, *rel.split("/"))


def main():
    frames = []
    for csv_file in (TRAIN_CSV, VALID_CSV):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        df = pd.read_csv(csv_file)
        df["Split"] = os.path.splitext(os.path.basename(csv_file))[0]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from train+valid CSVs")

    # Keep only rows whose image file actually exists
    df["image_path"] = df["Path"].map(resolve_path)
    df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)
    print(f"{len(df)} images exist on disk")

    # For each label, sample images where that label is positive (== 1.0)
    selected = []
    for label in LABELS:
        pos = df[df[label] == 1.0]
        n = min(IMAGES_PER_LABEL, len(pos))
        picked = pos.sample(n=n, random_state=RANDOM_SEED)
        selected.append(picked)
        print(f"  {label}: {n} images (of {len(pos)} positive)")

    subset = pd.concat(selected).drop_duplicates(subset=["Path"]).reset_index(drop=True)

    # Keep useful columns: resolved path, split, the 5 labels
    out = subset[["image_path", "Split"] + LABELS].copy()
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone: {len(out)} unique images -> {OUTPUT_CSV}")
    print(out.head())


if __name__ == "__main__":
    main()
