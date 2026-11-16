"""Create and verify a patient-disjoint 80/20 subset; never run training."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def value(text):
    return None if text.strip() == "" else float(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    output = args.output_dir.resolve()
    destinations = [output / "train_80.csv", output / "validation_20.csv", output / "verification.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Subset outputs already exist; choose a new output directory.")

    trainer = Path(__file__).resolve().parents[1] / "train_student.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    label_assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "LABELS" for t in n.targets))
    labels = ast.literal_eval(label_assignment.value)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fix_path")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(trainer), "exec"), namespace)
    fix_path = namespace["fix_path"]

    source = root / "train.csv"
    reserved = root / "valid.csv"
    original_hashes = {str(path): sha256(path) for path in (source, reserved)}
    fields, original = read_csv(source)
    metadata = {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}
    assert {"Path", "Frontal/Lateral", *labels} <= set(fields)
    other_findings = [c for c in fields if c not in metadata and c not in labels]

    def eligible(row):
        return (row["Frontal/Lateral"] == "Frontal"
                and any(value(row[c]) == 1 for c in labels)
                and not any(value(row[c]) == 1 for c in other_findings))

    pool = sorted((row for row in original if eligible(row)), key=lambda row: row["Path"])
    assert len({row["Path"] for row in pool}) == len(pool), "Duplicate eligible image paths"
    patients = np.array([re.search(r"patient\d+", row["Path"]).group() for row in pool])
    features = np.array([[value(row[c]) == state for state in (1, 0) for c in labels]
                         for row in pool], dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    available = np.ones(len(pool), dtype=bool)
    selected = {}
    goals_by_split = {"validation": [3] * 5 + [2] * 5, "train": [12] * 5 + [8] * 5}

    # Reserve validation coverage first. Patient removal applies globally, including
    # alternative studies/views for the same patient in the original training CSV.
    for split, size in (("validation", 20), ("train", 80)):
        counts = np.zeros(10, dtype=int)
        goals = np.array(goals_by_split[split])
        availability = features[available].sum(axis=0)
        assert np.all(availability > 0), (split, availability.tolist())
        rarity = 1 / np.sqrt(availability)
        rarity /= rarity.mean()
        indices = []
        for _ in range(size):
            deficits = np.maximum(goals - counts, 0)
            weights = (deficits > 0) * (1 + rarity) * deficits / np.maximum(goals, 1)
            scores = features @ weights
            scores[~available] = -np.inf
            best = np.flatnonzero(scores == scores.max())
            assert len(best), "Insufficient eligible patients"
            index = int(rng.choice(best))
            indices.append(index)
            counts += features[index]
            available[patients == patients[index]] = False
        selected[split] = [pool[i] for i in indices]
        assert np.all(counts > 0), (split, "Missing positive/negative coverage", counts.tolist())

    def patient_set(rows):
        return {re.search(r"patient\d+", row["Path"]).group() for row in rows}

    assert len(selected["train"]) == 80 and len(selected["validation"]) == 20
    assert len(patient_set(selected["train"])) == 80
    assert len(patient_set(selected["validation"])) == 20
    assert not patient_set(selected["train"]) & patient_set(selected["validation"])
    all_rows = selected["train"] + selected["validation"]
    assert len({row["Path"] for row in all_rows}) == 100
    source_by_path = {row["Path"]: row for row in original}
    resolved = set()
    for row in all_rows:
        assert eligible(row) and row == source_by_path[row["Path"]]
        path = fix_path(row["Path"], root).resolve()
        assert path.is_relative_to(root / "train") and path.is_file(), str(path)
        assert path not in resolved, str(path)
        resolved.add(path)
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            rgb.load()
    print("Verified 100 unique image files: all decoded and converted to RGB.", flush=True)

    report = {
        "seed": args.seed,
        "source_csv": str(source),
        "reserved_original_validation_csv": str(reserved),
        "data_dir": str(root),
        "source_sha256": original_hashes,
        "trainer_sha256": sha256(trainer),
        "eligible_images": len(pool),
        "eligible_patients": len(set(patients)),
        "excluded_if_positive": other_findings,
        "selection_method": "Validation first; greedy positive/negative coverage weighted by rarity and remaining deficit; seeded random tie-breaking; one image per patient globally.",
        "sampling_goals": goals_by_split,
        "goal_order": [f"{c}: {state}" for state in ("positive", "negative") for c in labels],
        "patient_overlap": 0,
        "unique_images_total": len(resolved),
        "all_100_images_opened": True,
        "original_values_preserved": True,
        "split_reports": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", destinations[0]), ("validation", destinations[1])):
        rows = selected[split]
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        saved_fields, saved_rows = read_csv(path)
        assert saved_fields == fields and saved_rows == rows, "CSV round trip changed values"
        counts = {c: {name: sum(value(row[c]) == state for row in rows)
                      for name, state in (("positive", 1), ("negative", 0), ("uncertain", -1), ("blank", None))}
                  for c in labels}
        assert all(sum(v.values()) == len(rows) for v in counts.values())
        report["split_reports"][split] = {
            "csv": str(path), "sha256": sha256(path), "images": len(rows),
            "patients": len(patient_set(rows)), "images_opened": len(rows),
            "label_counts": counts,
        }
    assert all(sha256(Path(path)) == digest for path, digest in original_hashes.items())
    report["original_source_csvs_unchanged"] = True
    destinations[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
