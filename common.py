"""Shared paths, label loading, image preprocessing, and cache keys.

Example:
    from common import load_labels, canonical_key, aspect_fit, to_tensor
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent          # Project-detail/
DATA_DIR = BASE / "Dataset"
SCRIPTS_DIR = BASE / "Scripts"
CACHE_DIR = SCRIPTS_DIR / "cache"
RUNS_DIR = SCRIPTS_DIR / "runs"

TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
NUM_CLASSES = len(TASKS)
INPUT_H, INPUT_W = 320, 384                     # H x W after aspect-fit + zero padding
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PATH_PREFIXES = {"CheXpert-v1.0-small", "CheXpert-v1.0", "Dataset"}


def canonical_key(csv_path) -> str:
    """'CheXpert-v1.0-small/train/patientX/studyY/view1_frontal.jpg'
    -> 'train/patientX/studyY/view1_frontal.jpg'. Forward slashes, no prefix.
    Every cache file and every lookup uses this key."""
    parts = str(csv_path).replace("\\", "/").strip().split("/")
    while parts and parts[0] in _PATH_PREFIXES:
        parts = parts[1:]
    return "/".join(parts)


def resolve_image(csv_path) -> Path:
    """CSV Path string -> absolute Path under DATA_DIR."""
    return DATA_DIR.joinpath(*canonical_key(csv_path).split("/"))


def load_labels(csv, blank_policy="zero", uncertain_policy="mask"):
    """Read a CheXpert-format CSV, keep frontal views, build targets/mask.

    Returns (df, targets[N,5] float32, mask[N,5] float32).
    blank_policy: "zero" (NaN -> negative, mask 1; CheXpert convention) or
                  "mask" (NaN excluded).
    uncertain_policy (-1): "mask" (default), "ones", "zeros".
    """
    df = pd.read_csv(csv)
    missing = {"Path", "Frontal/Lateral", *TASKS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {Path(csv).name}: {sorted(missing)}")
    df = df[df["Frontal/Lateral"].eq("Frontal")].copy()
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    if df["patient_id"].isna().any():
        raise ValueError("Could not extract patient IDs from some Paths")
    df = df.reset_index(drop=True)

    raw = df[TASKS].to_numpy(dtype=np.float32)          # may contain NaN / -1
    targets = np.zeros_like(raw)
    mask = np.ones_like(raw)
    valid = np.isin(raw, [0.0, 1.0])

    neg = np.isnan(raw)
    if blank_policy == "zero":
        mask[neg] = 1.0          # blank -> target 0 (already), keep mask
    elif blank_policy == "mask":
        mask[neg] = 0.0
    else:
        raise ValueError(f"blank_policy must be 'zero' or 'mask', got {blank_policy!r}")

    unc = raw == -1.0
    if uncertain_policy == "mask":
        mask[unc] = 0.0
    elif uncertain_policy == "ones":
        targets[unc] = 1.0
    elif uncertain_policy == "zeros":
        targets[unc] = 0.0
    else:
        raise ValueError(f"uncertain_policy must be mask/ones/zeros, got {uncertain_policy!r}")

    targets[valid] = raw[valid]
    # anything not 0/1/-1/NaN would already be excluded by `valid`; ensure mask
    mask[~valid & ~neg & ~unc] = 0.0

    targets = targets.astype(np.float32)
    mask = mask.astype(np.float32)

    for i, name in enumerate(TASKS):
        pos = int(((targets[:, i] == 1) & (mask[:, i] == 1)).sum())
        negn = int(((targets[:, i] == 0) & (mask[:, i] == 1)).sum())
        msk = int((mask[:, i] == 0).sum())
        print(f"  {name:18} pos={pos:3d} neg={negn:3d} masked={msk:3d}")
    return df, targets, mask


def patient_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Patient-disjoint split: shuffle patients, fill val until ~val_frac of images."""
    rng = random.Random(seed)
    patients = df["patient_id"].unique().tolist()
    rng.shuffle(patients)
    counts = df.groupby("patient_id").size().to_dict()
    n_total = len(df)
    val_patients, n_val = set(), 0
    for pid in patients:
        if n_val >= val_frac * n_total and n_val > 0:
            break
        val_patients.add(pid)
        n_val += counts[pid]
    is_val = df["patient_id"].isin(val_patients)
    return (df[~is_val].reset_index(drop=True),
            df[is_val].reset_index(drop=True))


# ---------------- Preprocessing / augmentation ----------------
def aspect_fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize keeping aspect ratio to fit h x w, zero-pad the short side.
    Identical geometry to train_student.py."""
    H, W = img.shape[:2]
    scale = min(h / H, w / W)
    newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), dtype=img.dtype)
    y0, x0 = (h - newH) // 2, (w - newW) // 2
    canvas[y0:y0 + newH, x0:x0 + newW] = resized
    return canvas


def aspect_fit_float(map_: np.ndarray, h: int, w: int, pad_value: float = 0.0):
    """Same geometry as aspect_fit but for float maps.

    Returns (canvas[h,w] float32, valid[h,w] float32) where valid is 1 inside
    the pasted (real image) region and 0 in the padding."""
    H, W = map_.shape[:2]
    scale = min(h / H, w / W)
    newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(map_.astype(np.float32), (newW, newH), interpolation=interp)
    canvas = np.full((h, w), pad_value, dtype=np.float32)
    if map_.ndim == 3:
        canvas = np.full((h, w, map_.shape[2]), pad_value, dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.float32)
    y0, x0 = (h - newH) // 2, (w - newW) // 2
    canvas[y0:y0 + newH, x0:x0 + newW] = resized
    valid[y0:y0 + newH, x0:x0 + newW] = 1.0
    return canvas, valid


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation ±10 deg, small scale/translate, brightness/contrast. NO hflip."""
    H, W = img.shape
    scale = rng.uniform(0.9, 1.1)
    angle = rng.uniform(-10, 10)
    tx = rng.uniform(-0.05, 0.05) * W
    ty = rng.uniform(-0.05, 0.05) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = rng.uniform(0.8, 1.2)   # contrast
    beta = rng.uniform(-30, 30)      # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def to_tensor(img: np.ndarray):
    """uint8 grayscale [H,W] -> float tensor [3,H,W], ImageNet normalized."""
    import torch
    img = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))
