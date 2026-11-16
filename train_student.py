#!/usr/bin/env python3
"""CheXpert-v1.0-small, 100-image, two-epoch student CNN smoke test.

Example (from Project-detail/Scripts on Windows):
    python train_v1.py --data-dir "../Dataset"

This is a pipeline test, not a statistically meaningful model evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

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
MODEL_OPTIONS = ("mobilenet_v3_small", "mobilenetv4_small", "efficientnet_b0")
MOBILENETV4_ID = "mobilenetv4_conv_small.e2400_r224_in1k"


def build_model(model_name, *, pretrained):
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
        return model
    if model_name == "mobilenetv4_small":
        import timm
        return timm.create_model(MOBILENETV4_ID, pretrained=pretrained, num_classes=len(LABELS))
    raise ValueError(f"Unknown model option: {model_name}")


def model_details(model_name, pretrained):
    if model_name == "mobilenet_v3_small":
        return {"architecture": "mobilenet_v3_small", "model_option": model_name,
                "model_backend": "torchvision", "model_identifier": "mobilenet_v3_small",
                "initialization": str(models.MobileNet_V3_Small_Weights.DEFAULT) if pretrained else "random",
                "torchvision_version": version("torchvision")}
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        return {"architecture": "efficientnet_b0", "model_option": model_name,
                "model_backend": "torchvision", "model_identifier": "efficientnet_b0",
                "initialization": f"{weights}; newly initialized five-output classifier" if pretrained else "random",
                "pretrained_weights": str(weights) if pretrained else None,
                "weights_url": weights.url if pretrained else None,
                "torchvision_version": version("torchvision")}
    if model_name == "mobilenetv4_small":
        return {"architecture": "mobilenetv4_conv_small", "model_option": model_name,
                "model_backend": "timm", "model_identifier": MOBILENETV4_ID,
                "timm_version": version("timm"), "torchvision_version": version("torchvision"),
                "initialization": "ImageNet-1k pretrained backbone; newly initialized five-output classifier" if pretrained else "random"}
    raise ValueError(f"Unknown model option: {model_name}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CheXpert student smoke test (100 frontal images)")
    p.add_argument("--data-dir", type=Path, default=Path("../Dataset"), help="Folder containing train.csv, valid.csv, train/, valid/")
    p.add_argument("--train-csv", type=Path, help="Use this training CSV directly; requires --valid-csv")
    p.add_argument("--valid-csv", type=Path, help="Use this validation CSV directly; requires --train-csv")
    p.add_argument("--full-data", action="store_true", help="Use all frontal rows from data-dir/train.csv and valid.csv without pilot sampling")
    p.add_argument("--log-every", type=int, default=0, help="Print and persist progress every N batches (0 disables)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--model", choices=MODEL_OPTIONS, default="mobilenet_v3_small")
    p.add_argument("--output-dir", type=Path, default=Path("../outputs/train_v1"))
    p.add_argument("--train-images", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="0 is reliable on Windows")
    p.add_argument("--no-pretrained", action="store_true", help="Use random weights if pretrained download unavailable")
    args = p.parse_args(argv)
    if (args.train_csv is None) != (args.valid_csv is None):
        p.error("--train-csv and --valid-csv must be supplied together")
    if args.full_data and args.train_csv is not None:
        p.error("--full-data uses the original CSVs in --data-dir; omit CSV overrides")
    if args.log_every < 0:
        p.error("--log-every must be nonnegative")
    return args


def fix_path(value: str, root: Path) -> Path:
    """CSV path is CheXpert-v1.0-small/train/patient... but root is Dataset/."""
    text = str(value).replace("\\", "/").strip()
    parts = text.split("/")
    if parts and parts[0] in {"CheXpert-v1.0-small", "Dataset"}:
        parts = parts[1:]
    if not parts or parts[0] not in {"train", "valid"}:
        raise ValueError(f"Unexpected image path in CSV: {value!r}")
    return root.joinpath(*parts)


def load_frontal(csv_path: Path, *, exact=False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", *LABELS}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(absent)}")
    if exact and not df["Frontal/Lateral"].eq("Frontal").all():
        raise ValueError(f"Explicit CSV must contain only frontal images: {csv_path}")
    df = df.loc[df["Frontal/Lateral"].eq("Frontal")].copy()
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    if df["patient_id"].isna().any():
        raise ValueError("Could not extract patient IDs from some Paths")
    return df.reset_index(drop=True)


def load_splits(args, data_dir):
    if args.full_data:
        train_csv, valid_csv = data_dir / "train.csv", data_dir / "valid.csv"
        train, valid = load_frontal(train_csv), load_frontal(valid_csv)
    elif args.train_csv is not None:
        train_csv = args.train_csv.expanduser().resolve()
        valid_csv = args.valid_csv.expanduser().resolve()
        train = load_frontal(train_csv, exact=True)
        valid = load_frontal(valid_csv, exact=True)
    else:
        train_csv, valid_csv = data_dir / "train.csv", data_dir / "valid.csv"
        train = choose_pilot(load_frontal(train_csv), args.train_images, args.seed)
        valid = load_frontal(valid_csv)
    for name, frame in (("train", train), ("valid", valid)):
        if frame.empty or frame["Path"].duplicated().any():
            raise ValueError(f"{name} CSV must contain nonempty, unique image paths")
    if set(train.patient_id) & set(valid.patient_id):
        raise ValueError("Patient leakage across train and valid!")
    return train, valid, train_csv, valid_csv


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


def run_epoch(model, loader, device, optimizer=None, *, log_every=0, phase="", progress_path=None):
    started = time.perf_counter()
    training = optimizer is not None
    model.train(training)
    loss_num = 0.0
    correct = 0.0
    valid_total = 0.0
    images_seen = batches_seen = 0
    last_batch_images = 0
    for images, targets, mask in loader:
        images = images.to(device)
        targets, mask = targets.to(device), mask.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            if device.type == "cuda" and not logits.is_cuda:
                raise RuntimeError("CUDA was requested but model logits are not on CUDA")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite model logits")
            loss = masked_loss(logits, targets, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            if training and mask.sum().item() > 0:
                loss.backward()
                optimizer.step()
        observed_count = mask.sum().item()
        loss_num += loss.item() * observed_count
        predictions = (logits.sigmoid() >= 0.5).float()
        correct += ((predictions == targets).float() * mask).sum().item()
        valid_total += observed_count
        images_seen += images.shape[0]
        batches_seen += 1
        last_batch_images = len(images)
        if log_every and (batches_seen % log_every == 0 or batches_seen == len(loader)):
            progress = {"phase": phase, "device": str(device), "batches": batches_seen,
                        "expected_batches": len(loader), "images": images_seen,
                        "expected_images": len(loader.dataset), "observed_labels": int(valid_total),
                        "loss": loss_num / max(valid_total, 1), "finite_losses": True,
                        "elapsed_seconds": time.perf_counter() - started}
            print(f"{phase} | batches {batches_seen}/{len(loader)} | images {images_seen}/{len(loader.dataset)} | "
                  f"loss {progress['loss']:.6f} | elapsed {progress['elapsed_seconds']:.1f}s", flush=True)
            if progress_path is not None:
                progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    if images_seen != len(loader.dataset):
        raise RuntimeError(f"Processed {images_seen} images; expected {len(loader.dataset)}")
    return {"loss": loss_num / max(valid_total, 1), "accuracy": correct / max(valid_total, 1),
            "images": images_seen, "batches": batches_seen, "observed_labels": int(valid_total),
            "correct_labels": int(correct), "last_batch_images": last_batch_images}


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
    run_started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    if args.train_images <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--train-images, --epochs and --batch-size must be positive")
    data_dir = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    print("Dataset:", data_dir)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    pilot, valid, train_csv, valid_csv = load_splits(args, data_dir)
    print("Training CSV:", train_csv)
    print("Validation CSV:", valid_csv)
    print("CSV overrides used directly:", args.train_csv is not None)
    print("Full data without pilot sampling:", args.full_data)
    print(f"Expected images: {len(pilot)} train / {len(valid)} valid; "
          f"expected final batches: {(len(pilot) - 1) % args.batch_size + 1} train / "
          f"{(len(valid) - 1) % args.batch_size + 1} valid", flush=True)

    # Check paths before initializing/downloading pretrained weights.
    for part_name, frame in (("train", pilot), ("valid", valid)):
        unresolved = [fix_path(p, data_dir) for p in frame["Path"] if not fix_path(p, data_dir).is_file()]
        if unresolved:
            raise FileNotFoundError(
                f"{len(unresolved)} {part_name} image files not found; first: {unresolved[0]}\n"
                f"Expected image directories: {data_dir / 'train'} and {data_dir / 'valid'}"
            )
    print(f"Image path check passed: {len(pilot)} training and {len(valid)} validation images.")
    if args.full_data:
        pilot.to_csv(output / "train_used.csv", index=False)
        valid.to_csv(output / "valid_used.csv", index=False)
    elif args.train_csv is not None:
        shutil.copyfile(train_csv, output / "train_used.csv")
        shutil.copyfile(valid_csv, output / "valid_used.csv")
    else:
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

    details = model_details(args.model, not args.no_pretrained)
    print(f"\nLoading {details['model_identifier']} ({details['initialization']})...")
    try:
        model = build_model(args.model, pretrained=not args.no_pretrained)
    except Exception as e:
        raise RuntimeError(f"Could not load {args.model}. Ensure timm is installed for MobileNetV4 and internet is available for first run, "
                           "or add --no-pretrained for a random-initialized smoke test.") from e
    model = model.to(device)
    print("Model parameter device:", next(model.parameters()).device, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"Model outputs: {len(LABELS)} independent disease logits")
    print("Accuracy = fraction of observed disease labels predicted correctly; not per-image accuracy.")
    print(f"Training run: {len(pilot)} training / {len(valid)} validation images, {args.epochs} epochs.\n")

    metadata = {
        **details, "labels": LABELS,
        "seed": args.seed, "pretrained": not args.no_pretrained,
        "full_data": args.full_data, "configured_epochs": args.epochs,
        "settings_origin": "Trainer defaults; README.md is empty and no separate full-run schedule is documented" if args.full_data else "CLI/default settings",
        "run_started_utc": run_started_utc, "optimizer": "AdamW", "weight_decay": 1e-4,
        "row_filter": "Frontal views only; no positive-only requirement or other-finding exclusion",
        "train_image_count": len(pilot), "val_image_count": len(valid),
        "train_patient_count": int(pilot.patient_id.nunique()),
        "val_patient_count": int(valid.patient_id.nunique()), "patient_overlap": 0,
        "train_csv": str(train_csv), "valid_csv": str(valid_csv), "data_dir": str(data_dir),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size, "num_workers": args.num_workers, "lr": args.lr,
        "train_observed_labels": int(train_ds.mask.sum().item()),
        "val_observed_labels": int(val_ds.mask.sum().item()),
        "pretrained_source": getattr(model, "pretrained_cfg", {}).get("hf_hub_id"),
        "preprocessing": {"resize": [224, 224], "mode": "RGB",
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "label_policy": "Only exact 0 and 1 are observed; uncertain -1 and blank/NaN are masked from loss and accuracy.",
        "accuracy": "Correct observed binary labels / observed labels; sigmoid threshold >= 0.5. Not per-image accuracy.",
        "loss": "Mean binary cross entropy with logits over observed labels.",
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Expected observed labels: {metadata['train_observed_labels']} train / {metadata['val_observed_labels']} valid", flush=True)

    history = []
    best_val = float("inf")
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_stats = run_epoch(model, train_loader, device, optimizer, log_every=args.log_every,
                                phase=f"Epoch {epoch}/{args.epochs} train", progress_path=output / "progress.json")
        with torch.inference_mode():
            valid_stats = run_epoch(model, val_loader, device, log_every=args.log_every,
                                    phase=f"Epoch {epoch}/{args.epochs} valid", progress_path=output / "progress.json")
        for prefix, stats in (("train", train_stats), ("val", valid_stats)):
            if stats["observed_labels"] != metadata[f"{prefix}_observed_labels"]:
                raise RuntimeError(f"Observed-label coverage mismatch in {prefix}: {stats}")
            expected_last = (metadata[f"{prefix}_image_count"] - 1) % args.batch_size + 1
            if stats["last_batch_images"] != expected_last:
                raise RuntimeError(f"Final batch coverage mismatch in {prefix}: {stats}")
        tr_loss, tr_acc = train_stats["loss"], train_stats["accuracy"]
        va_loss, va_acc = valid_stats["loss"], valid_stats["accuracy"]
        record = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                  "val_loss": va_loss, "val_acc": va_acc,
                  **{f"train_{k}": v for k, v in train_stats.items() if k not in ("loss", "accuracy")},
                  **{f"val_{k}": v for k, v in valid_stats.items() if k not in ("loss", "accuracy")},
                  "epoch_seconds": time.perf_counter() - epoch_started,
                  "elapsed_training_seconds": time.perf_counter() - training_started}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} | train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
              f"val loss: {va_loss:.4f} acc: {va_acc:.3f} | "
              f"images: {train_stats['images']} train / {valid_stats['images']} valid | "
              f"batches: {train_stats['batches']} train / {valid_stats['batches']} valid | "
              f"observed labels: {train_stats['observed_labels']} train / {valid_stats['observed_labels']} valid", flush=True)
        print(f"Final batch images: {train_stats['last_batch_images']} train / {valid_stats['last_batch_images']} valid; "
              f"epoch elapsed: {record['epoch_seconds']:.1f}s", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            atomic_torch_save({**metadata, "state_dict": model.state_dict(), "epoch": epoch,
                        "validation_loss": va_loss, "metrics": record},
                       output / "student_best.pth")

        atomic_torch_save({**metadata, "state_dict": model.state_dict(), "epoch": epoch,
                           "metrics": record}, output / "student_last.pth")
        atomic_torch_save(model.state_dict(), output / "student_weights.pth")
        temporary_history = output / "history.csv.tmp"
        with temporary_history.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        temporary_history.replace(output / "history.csv")
        save_curves(history, output / "learning_curves.png")
        summary = {"completed_epochs": epoch, "configured_epochs": args.epochs,
                   "best_epoch": min(history, key=lambda r: r["val_loss"])["epoch"],
                   "best_metrics": min(history, key=lambda r: r["val_loss"]), "final_metrics": record,
                   "training_elapsed_seconds": time.perf_counter() - training_started,
                   "total_elapsed_seconds": time.perf_counter() - run_started,
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nSaved to:", output)
    for filename in ("student_best.pth", "student_last.pth", "student_weights.pth", "history.csv", "learning_curves.png", "run_config.json"):
        print(" -", filename)
    if not args.full_data:
        print("NOTE: Small pilot is deliberately class-aware and not representative of the natural distribution.")


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
