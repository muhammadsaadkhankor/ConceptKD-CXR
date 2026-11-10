#!/usr/bin/env python3
"""Read-only analysis of CheXpert-v1.0-small train.csv and valid.csv.

Usage:
    python dataset_analyze.py --data-dir /path/to/CheXpert-v1.0-small

Requires: pandas (`pip install pandas`). No model training or file modifications.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Original CheXpert observation columns (14 labels, including No Finding and Support Devices).
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
PROJECT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
]


def locate_csvs(data_dir: Path) -> Path:
    candidates = [data_dir, data_dir / "CheXpert-v1.0-small"]
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "valid.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find train.csv and valid.csv in {data_dir} or its CheXpert-v1.0-small subfolder. "
        "Pass the dataset directory with --data-dir."
    )


def print_label_counts(df: pd.DataFrame, labels: list[str]) -> None:
    results = []
    for label in labels:
        if label not in df.columns:
            continue
        values = pd.to_numeric(df[label], errors="coerce")
        results.append({
            "Label": label,
            "Positive (+1)": int(values.eq(1).sum()),
            "Negative (0)": int(values.eq(0).sum()),
            "Uncertain (-1)": int(values.eq(-1).sum()),
            "Missing": int(values.isna().sum()),
            "Other": int((values.notna() & ~values.isin([-1, 0, 1])).sum()),
        })
    if results:
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("No expected CheXpert label columns found.")


def analyze_split(name: str, csv_path: Path, image_root: Path, check_paths: int) -> None:
    print(f"\n{'=' * 90}\n{name.upper()} — {csv_path}\n{'=' * 90}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Rows (image records): {len(df):,}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")

    if "Path" in df:
        patient_ids = df["Path"].astype(str).str.extract(r"(patient\d+)", expand=False)
        print(f"Unique patients: {patient_ids.nunique():,}")
        print(f"Missing patient IDs in paths: {patient_ids.isna().sum():,}")
        studies = df["Path"].astype(str).str.extract(r"(patient\d+/study\d+)", expand=False)
        print(f"Unique patient studies: {studies.nunique():,}")
    else:
        print("WARNING: Missing 'Path' column.")

    for col in ("Frontal/Lateral", "AP/PA", "Sex", "Age"):
        if col in df:
            print(f"\n{col}:")
            if col == "Age":
                print(pd.to_numeric(df[col], errors="coerce").describe().to_string())
            else:
                print(df[col].fillna("<missing>").value_counts(dropna=False).to_string())

    print("\nALL CHEXPERT LABELS (counts are IMAGE counts, not exclusive classes):")
    print_label_counts(df, LABELS)

    selected = [lab for lab in PROJECT_LABELS if lab in df.columns]
    print("\nOUR 5 PROJECT LABELS:")
    print_label_counts(df, PROJECT_LABELS)

    if selected:
        numeric = df[selected].apply(pd.to_numeric, errors="coerce")
        num_positive = numeric.eq(1).sum(axis=1)
        print("\nNumber of SELECTED positive labels per image:")
        print(num_positive.value_counts().sort_index().rename_axis("Positive labels").to_string())
        print(f"Images positive for >=1 project label: {num_positive.ge(1).sum():,}")
        print(f"Images positive for 0 project labels: {num_positive.eq(0).sum():,}")
        print(f"Images with all 5 project labels explicitly negative: {numeric.eq(0).all(axis=1).sum():,}")
        print(f"Images with >=1 unknown/missing project label: {(numeric.eq(-1) | numeric.isna()).any(axis=1).sum():,}")

        print("\nCo-occurrence of POSITIVE project labels (diagonal = positive count):")
        positive = numeric.eq(1).astype(int)
        print(positive.T.dot(positive).to_string())

        if "Frontal/Lateral" in df:
            frontal = df["Frontal/Lateral"].astype(str).eq("Frontal")
            print(f"\nFrontal images only: {frontal.sum():,}")
            print("Selected label statistics for FRONTAL images:")
            print_label_counts(df.loc[frontal], PROJECT_LABELS)

    if "Path" in df and check_paths > 0:
        sample = df["Path"].dropna().astype(str).head(check_paths)
        missing = []
        for rel in sample:
            p = Path(rel)
            candidates = [p] if p.is_absolute() else [image_root / p, image_root.parent / p]
            if not any(path.is_file() for path in candidates):
                missing.append(rel)
        print(f"\nImage path check: {len(sample) - len(missing)}/{len(sample)} first CSV paths found")
        if missing:
            print("Example unresolved path:", missing[0])
            print("Expected layout: <parent>/CheXpert-v1.0-small/train/... or valid/...")

    print("\nFirst 3 CSV rows (selected fields):")
    show_cols = [c for c in ["Path", "Frontal/Lateral", "AP/PA", *PROJECT_LABELS] if c in df]
    print(df[show_cols].head(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CheXpert CSVs without training.")
    parser.add_argument("--data-dir", type=Path, default=Path("."),
                        help="Path containing train.csv/valid.csv or CheXpert-v1.0-small/ (default: current directory)")
    parser.add_argument("--check-paths", type=int, default=20,
                        help="Number of image paths to verify in each split (default: 20; use 0 to skip)")
    args = parser.parse_args()

    root = locate_csvs(args.data_dir.expanduser().resolve())
    print(f"Dataset directory: {root}")
    print("Read-only analysis. No training and no files will be written.")
    for split in ("train", "valid"):
        analyze_split(split, root / f"{split}.csv", root, max(0, args.check_paths))
    print("\nAnalysis complete. Note: CheXpert is MULTI-LABEL; images may have multiple positive findings.")


if __name__ == "__main__":
    main()
