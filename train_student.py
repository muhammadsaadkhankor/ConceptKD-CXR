#!/usr/bin/env python3
"""CheXpert-v1.0-small, 100-image, two-epoch student CNN smoke test.

Example (from Project-detail/Scripts on Windows):
    python train_v1.py --data-dir "../Dataset"

This is a pipeline test, not a statistically meaningful model evaluation.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def parse_args():
    p = argparse.ArgumentParser(description="CheXpert student smoke test (100 frontal images)")
    p.add_argument("--data-dir", type=Path, default=Path("../Dataset"), help="Folder containing train.csv, valid.csv, train/, valid/")
    p.add_argument("--output-dir", type=Path, default=Path("../outputs/train_v1"))
    p.add_argument("--train-images", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="0 is reliable on Windows")
    p.add_argument("--no-pretrained", action="store_true", help="Use random weights if pretrained download unavailable")
    return p.parse_args()


def fix_path(value: str, root: Path) -> Path:
    """CSV path is CheXpert-v1.0-small/train/patient... but root is Dataset/."""
    text = str(value).replace("\\", "/").strip()
    parts = text.split("/")
    if parts and parts[0] in {"CheXpert-v1.0-small", "Dataset"}:
        parts = parts[1:]
    if not parts or parts[0] not in {"train", "valid"}:
        raise ValueError(f"Unexpected image path in CSV: {value!r}")
    return root.joinpath(*parts)


def load_frontal(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", *LABELS}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(absent)}")
    df = df.loc[df["Frontal/Lateral"].eq("Frontal")].copy()
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    if df["patient_id"].isna().any():
        raise ValueError("Could not extract patient IDs from some Paths")
    return df.reset_index(drop=True)


def choose_pilot(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Greedy class-aware sampling to cover observed positives AND negatives.

    This is intentionally not a random representative sample; designed for testing
    the multi-label training pipeline. Choose one image per patient.
    """
    rng = np.random.default_rng(seed)
    # One image per patient to limit duplicates and avoid selection from repeated exams.
    eligible = df.sample(frac=1, random_state=seed).drop_duplicates("patient_id").reset_index(drop=True)
    if len(eligible) < n:
        raise ValueError(f"Requested {n} images but only {len(eligible)} unique patients available")
    # Select a sufficiently large candidate pool while keeping greedy search fast.
    pool = eligible.iloc[: min(len(eligible), 30000)].copy()
    observed = pool[LABELS].to_numpy(dtype=float)
    pos = observed == 1.0
    neg = observed == 0.0
    # Bias towards rare explicit negatives, e.g. atelectasis, while retaining positives.
    required_pos = min(15, max(1, n // 7))
    required_neg = min(10, max(1, n // 10))
    goals = np.array([required_pos] * 5 + [required_neg] * 5, dtype=int)
    # Pre-check whether sampled candidates have each target.
    availability = np.r_[pos.sum(axis=0), neg.sum(axis=0)]
    if np.any(availability < goals):
        print("WARNING: candidate pool lacks some target labels; sampler will do its best.")
    feature = np.concatenate([pos, neg], axis=1).astype(np.int8)
    counts = np.zeros(10, dtype=int)
    used = np.zeros(len(pool), dtype=bool)
    selected = []
    rare_weight = 1 / np.sqrt(np.maximum(availability, 1))
    rare_weight = rare_weight / rare_weight.mean()

    for _ in range(n):
        deficits = np.maximum(goals - counts, 0)
        if deficits.sum() > 0:
            # Prioritize outstanding goals, with extra value for scarcer evidence.
            weighted = (deficits > 0) * (1.0 + 0.5 * rare_weight)
            scores = feature @ weighted
        else:
            # After coverage, sample remaining images without optimizing train metric.
            scores = np.zeros(len(pool), dtype=float)
        scores[used] = -1e9
        best = np.flatnonzero(scores == scores.max())
        idx = int(rng.choice(best))
        selected.append(idx)
        used[idx] = True
        counts += feature[idx]

    chosen = pool.iloc[selected].copy().reset_index(drop=True)
    print(f"\nSelected {len(chosen)} unique-patient frontal TRAIN images:")
    for label in LABELS:
        v = chosen[label]
        print(f"  {label:19s} +1: {(v == 1).sum():3d} | 0: {(v == 0).sum():3d} | -1: {(v == -1).sum():3d} | missing: {v.isna().sum():3d}")
    short = goals - counts
    if np.any(short > 0):
        print("NOTE: Not every positive/negative coverage goal was achieved; pipeline test remains possible.")
    return chosen


class CheXpertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = transform
        raw = self.frame[LABELS].to_numpy(dtype=np.float32)
        self.mask = torch.from_numpy(np.isin(raw, [0., 1.]).astype(np.float32))
        self.targets = torch.from_numpy(np.where(raw == 1., 1., 0.).astype(np.float32))
        self.paths = [fix_path(p, root) for p in self.frame["Path"]]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            # Grayscale chest x-ray -> 3 channels for pretrained ImageNet model.
            image = self.transform(im.convert("RGB"))
        return image, self.targets[idx], self.mask[idx]


def masked_loss(logits, targets, mask):
    per_label = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # Valid targets are precisely 0 and 1; -1 and missing are masked.
    return (per_label * mask).sum() / mask.sum().clamp_min(1)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_num = 0.0
    correct = 0.0
    valid_total = 0.0
    for images, targets, mask in loader:
        images = images.to(device)
        targets, mask = targets.to(device), mask.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = masked_loss(logits, targets, mask)
            if training and mask.sum().item() > 0:
                loss.backward()
                optimizer.step()
        observed_count = mask.sum().item()
        loss_num += loss.item() * observed_count
        predictions = (logits.sigmoid() >= 0.5).float()
        correct += ((predictions == targets).float() * mask).sum().item()
        valid_total += observed_count
    return (loss_num / max(valid_total, 1), correct / max(valid_total, 1))


def save_curves(history, path: Path):
    epochs = [r["epoch"] for r in history]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train")
    axs[0].plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation")
    axs[0].set(title="Masked BCE loss", xlabel="Epoch", ylabel="Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[1].plot(epochs, [r["train_acc"] for r in history], marker="o", label="Train")
    axs[1].plot(epochs, [r["val_acc"] for r in history], marker="o", label="Validation")
    axs[1].set(title="Observed-label accuracy (threshold 0.5)", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    axs[0].set_xticks(epochs)
    axs[1].set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.train_images <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--train-images, --epochs and --batch-size must be positive")
    data_dir = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "valid.csv"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"Expected {data_dir / f}. Use --data-dir ../Dataset")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dataset:", data_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    full_train = load_frontal(data_dir / "train.csv")
    valid = load_frontal(data_dir / "valid.csv")
    pilot = choose_pilot(full_train, args.train_images, args.seed)
    assert not (set(pilot.patient_id) & set(valid.patient_id)), "Patient leakage across train and valid!"

    # Check paths before initializing/downloading pretrained weights.
    for part_name, frame in (("train", pilot), ("valid", valid)):
        unresolved = [fix_path(p, data_dir) for p in frame["Path"] if not fix_path(p, data_dir).is_file()]
        if unresolved:
            raise FileNotFoundError(
                f"{len(unresolved)} {part_name} image files not found; first: {unresolved[0]}\n"
                f"Expected image directories: {data_dir / 'train'} and {data_dir / 'valid'}"
            )
    print(f"Image path check passed: {len(pilot)} training and {len(valid)} validation images.")
    pilot.to_csv(output / "pilot_100.csv", index=False)

    # Pretrained ImageNet normalization, no lateral images and no augmentations in smoke test.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = CheXpertDataset(pilot, data_dir, transform)
    val_ds = CheXpertDataset(valid, data_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print("\nLoading MobileNetV3-Small (ImageNet pretrained)..." if not args.no_pretrained
          else "\nLoading MobileNetV3-Small (random initialization)...")
    try:
        weights = None if args.no_pretrained else models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as e:
        raise RuntimeError("Could not load pretrained weights. Ensure internet is available for first run, "
                           "or add --no-pretrained for a random-initialized smoke test.") from e
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"Model outputs: {len(LABELS)} independent disease logits")
    print("Accuracy = fraction of observed disease labels predicted correctly; not per-image accuracy.")
    print("Important: Only 100 training images and 2 epochs: results are for debugging, NOT model quality.\n")

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                  "val_loss": va_loss, "val_acc": va_acc}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} | train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
              f"val loss: {va_loss:.4f} acc: {va_acc:.3f}", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": model.state_dict(), "architecture": "mobilenet_v3_small",
                        "labels": LABELS, "epoch": epoch, "seed": args.seed,
                        "train_image_count": len(pilot), "val_image_count": len(valid),
                        "validation_loss": va_loss, "pretrained": not args.no_pretrained},
                       output / "student_best.pth")

    torch.save({"state_dict": model.state_dict(), "architecture": "mobilenet_v3_small",
                "labels": LABELS, "epoch": args.epochs, "seed": args.seed,
                "pretrained": not args.no_pretrained}, output / "student_last.pth")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, output / "learning_curves.png")
    print("\nSaved to:", output)
    for filename in ("pilot_100.csv", "student_best.pth", "student_last.pth", "history.csv", "learning_curves.png"):
        print(" -", filename)
    print("NOTE: Small pilot is deliberately class-aware and not representative of the natural distribution.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
