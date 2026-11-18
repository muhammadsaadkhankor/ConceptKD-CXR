#!/usr/bin/env python3
"""CheXpert knowledge-distillation: student training script.

Debug-scale run: ~87 frontal images, cached teacher logits, 8 GB GPU.

Example (from Project-detail/Scripts):
    python train_student.py --mode B2 --epochs 30
    python train_student.py --overfit          # pipeline sanity check
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------- Config ----------------
BASE = Path(__file__).resolve().parent.parent          # Project-detail/
DATA_DIR = BASE / "Dataset"
CSV_PATH = DATA_DIR / "train_100.csv"
LOGITS_PATH = BASE / "Scripts" / "teacher_logits_convnext_tiny.pt"
CONCEPTS_PATH = BASE / "Scripts" / "concept_targets_biovilt.pt"
OUT_DIR = BASE / "outputs" / "student"

TASKS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
NUM_CLASSES = len(TASKS)
NUM_CONCEPTS = 10            # concept head exists now; loss weight 0 until targets exist
INPUT_H, INPUT_W = 320, 384  # H x W after aspect-fit + zero padding
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# -----------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="CheXpert KD student (debug scale)")
    p.add_argument("--mode", choices=["B1", "B2", "B3", "Ours", "C1"], default="B2",
                   help="B1=sup, B2=sup+KD, B3=sup+concept, Ours=sup+KD+concept, "
                        "C1=sup+KD+shuffled concepts")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--logits", type=Path, default=LOGITS_PATH)
    p.add_argument("--concepts", type=Path, default=CONCEPTS_PATH)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="0 is reliable on Windows")
    p.add_argument("--lambda-sup", type=float, default=1.0)
    p.add_argument("--lambda-kd", type=float, default=1.0)
    p.add_argument("--lambda-con", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--num-concepts", type=int, default=NUM_CONCEPTS)
    p.add_argument("--overfit", action="store_true",
                   help="Train+eval on same images, no augmentation, 30 epochs")
    return p.parse_args()


# ---------------- Data ----------------
def fix_path(value: str) -> Path:
    """'CheXpert-v1.0-small/train/patient...' -> DATA_DIR/train/patient..."""
    text = str(value).replace("\\", "/").strip()
    parts = text.split("/")
    if parts and parts[0] in {"CheXpert-v1.0-small", "CheXpert-v1.0", "Dataset"}:
        parts = parts[1:]
    if not parts or parts[0] not in {"train", "valid"}:
        raise ValueError(f"Unexpected image path in CSV: {value!r}")
    return DATA_DIR.joinpath(*parts)


def teacher_key(csv_path_str: str) -> str:
    """Convert original CSV Path to the key style used in the logits .pt file.

    The cached logits were keyed by confirm.py's image_path strings like
    '..\\Dataset\\train\\patientX\\studyY\\view1_frontal.jpg'.
    """
    text = str(csv_path_str).replace("\\", "/")
    parts = text.split("/")
    if parts and parts[0] in {"CheXpert-v1.0-small", "CheXpert-v1.0"}:
        parts = parts[1:]
    return ".." + os.sep + "Dataset" + os.sep + os.sep.join(parts)


def load_concepts(path: Path):
    """Load cached BioViL-T concept targets.

    Format (see cache_concpets.py):
        {"concepts": [names], "scores": {Path: tensor[C]},
         "maps": {Path: tensor[C,10,12]}, "map_valid": optional}
    Keys are the original CSV Path strings.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    concepts = blob["concepts"]
    map_hw = tuple(blob.get("map_hw", ()))
    if map_hw != (10, 12):
        raise ValueError(f"{path.name}: map_hw is {map_hw}, expected (10, 12). "
                         "Re-run cache_concepts.py.")
    scores = {k: v.float() for k, v in blob["scores"].items()}
    maps = {k: v.float() for k, v in blob["maps"].items()}
    if "map_valid" not in blob or not isinstance(blob["map_valid"], dict):
        raise ValueError(f"{path.name} has no per-image 'map_valid'. "
                         "Re-run cache_concepts.py.")
    mvalid = {k: v.float() for k, v in blob["map_valid"].items()}
    return {"names": concepts, "scores": scores, "maps": maps,
            "map_valid": mvalid}


def get_concept(concepts: dict, csv_path_str: str):
    """Return (score[C], map[C,10,12], map_valid[10,12]) for one image."""
    key = str(csv_path_str)
    if key not in concepts["scores"]:
        raise KeyError(f"No concept targets for {key!r}")
    score = concepts["scores"][key]
    cmap = concepts["maps"][key]
    mvalid = concepts["map_valid"][key]
    return score, cmap, mvalid


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"Path", "Frontal/Lateral", *TASKS} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")
    df = df[df["Frontal/Lateral"].eq("Frontal")].copy()
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    if df["patient_id"].isna().any():
        raise ValueError("Could not extract patient IDs from some Paths")
    return df.reset_index(drop=True)


def patient_split(df: pd.DataFrame, val_target: int = 17, seed: int = 42):
    """Greedy patient-disjoint split: ~70 train / ~17 val images."""
    rng = random.Random(seed)
    patients = df["patient_id"].unique().tolist()
    rng.shuffle(patients)
    counts = df.groupby("patient_id").size().to_dict()
    train_patients, n_train = set(), 0
    for pid in patients:
        if n_train + counts[pid] <= 70 or n_train == 0:
            train_patients.add(pid)
            n_train += counts[pid]
        if n_train >= 70:
            break
    is_train = df["patient_id"].isin(train_patients)
    return df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)


def load_teacher_logits(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tasks = blob["tasks"]
    if tasks != TASKS:
        raise ValueError(f"Teacher tasks {tasks} != expected {TASKS}")
    # Normalize keys to forward slashes for robust lookup
    logits = {k.replace("\\", "/"): v.float() for k, v in blob["logits"].items()}
    return logits


def get_teacher(logits: dict, csv_path_str: str) -> torch.Tensor:
    candidates = [
        csv_path_str.replace("\\", "/"),
        teacher_key(csv_path_str).replace("\\", "/"),
        str(fix_path(csv_path_str)).replace("\\", "/"),
    ]
    for c in candidates:
        if c in logits:
            return logits[c]
    raise KeyError(f"No teacher logits for {csv_path_str!r}")


# ---------------- Preprocessing / augmentation ----------------
def aspect_fit(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize keeping aspect ratio to fit h x w, zero-pad the short side."""
    H, W = img.shape[:2]
    scale = min(h / H, w / W)
    newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), dtype=np.uint8)
    y0, x0 = (h - newH) // 2, (w - newW) // 2
    canvas[y0:y0 + newH, x0:x0 + newW] = resized
    return canvas


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation ±10 deg, small scale/translate, brightness/contrast. NO hflip."""
    H, W = img.shape
    # scale 0.9-1.1, translate ±5% of size, rotate ±10 deg -> single affine
    scale = rng.uniform(0.9, 1.1)
    angle = rng.uniform(-10, 10)
    tx = rng.uniform(-0.05, 0.05) * W
    ty = rng.uniform(-0.05, 0.05) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # brightness/contrast jitter
    alpha = rng.uniform(0.8, 1.2)   # contrast
    beta = rng.uniform(-30, 30)      # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


class CheXpertDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, teacher_logits: dict,
                 concepts: dict = None, concept_perm: np.ndarray = None,
                 num_concepts: int = NUM_CONCEPTS,
                 augment_fn=None, seed: int = 42):
        self.frame = frame.reset_index(drop=True)
        self.augment_fn = augment_fn
        self.rng = np.random.default_rng(seed)
        self.paths = [fix_path(p) for p in self.frame["Path"]]
        self.concepts = concepts
        self.concept_perm = concept_perm   # C1: permuted indices into concept arrays
        self.num_concepts = num_concepts

        raw = self.frame[TASKS].to_numpy(dtype=np.float32)
        # mask: only 0.0 and 1.0 are valid; -1 (uncertain) and NaN masked out
        self.mask = torch.from_numpy(np.isin(raw, [0.0, 1.0]).astype(np.float32))
        self.targets = torch.from_numpy(np.where(raw == 1.0, 1.0, 0.0).astype(np.float32))
        # teacher soft targets, aligned by original Path string
        self.teacher = torch.stack([get_teacher(teacher_logits, p)
                                    for p in self.frame["Path"]])

        # concept targets (score[C], map[C,10,12], map_valid[10,12]); None if absent
        if concepts is not None:
            cs, cm, cv = [], [], []
            for p in self.frame["Path"]:
                s, m, v = get_concept(concepts, p)
                cs.append(s)
                cm.append(m)
                cv.append(v)
            self.c_score = torch.stack(cs)
            self.c_map = torch.stack(cm)
            self.c_valid = torch.stack(cv)
        else:
            z = torch.zeros(num_concepts)
            self.c_score = torch.stack([z] * len(self.frame))
            self.c_map = torch.zeros(len(self.frame), num_concepts, 10, 12)
            self.c_valid = torch.ones(len(self.frame), 10, 12)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.paths[idx]}")
        img = aspect_fit(img, INPUT_H, INPUT_W)
        if self.augment_fn is not None:
            img = self.augment_fn(img, self.rng)

        cidx = idx if self.concept_perm is None else int(self.concept_perm[idx])
        return (to_tensor(img), self.targets[idx], self.mask[idx],
                self.teacher[idx], self.c_score[cidx], self.c_map[cidx],
                self.c_valid[cidx])


# ---------------- Model ----------------
class Student(nn.Module):
    def __init__(self, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.encoder = timm.create_model("efficientnet_b0.ra_in1k",
                                         pretrained=True, num_classes=0)
        feat_dim = self.encoder.num_features          # 1280
        self.disease_head = nn.Linear(feat_dim, NUM_CLASSES)
        self.concept_head = nn.Linear(feat_dim, num_concepts)

    def forward(self, x):
        feat = self.encoder(x)                        # [B, 1280]
        return self.disease_head(feat), self.concept_head(feat)


# ---------------- Losses ----------------
def loss_sup(disease_logits, targets, mask):
    """Masked BCE over valid (0/1) entries only; -1 and NaN excluded."""
    per = F.binary_cross_entropy_with_logits(disease_logits, targets, reduction="none")
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def loss_kd(student_logits, teacher_logits, temperature):
    """Temperature-scaled per-label BCE on teacher soft targets."""
    s = torch.sigmoid(student_logits / temperature)
    t = torch.sigmoid(teacher_logits / temperature)
    return F.binary_cross_entropy(s, t, reduction="mean") * temperature ** 2


def loss_con(concept_logits, concept_scores):
    """BCE between sigmoid(concept_logits) and soft BioViL-T concept scores."""
    if concept_scores is None:
        return None
    return F.binary_cross_entropy(torch.sigmoid(concept_logits), concept_scores,
                                  reduction="mean")


def total_loss(disease_logits, concept_logits, targets, mask, teacher,
               lambdas, temperature, concept_scores=None):
    l_sup = loss_sup(disease_logits, targets, mask)
    l_kd = loss_kd(disease_logits, teacher, temperature)
    l_con = loss_con(concept_logits, concept_scores)
    total = lambdas["sup"] * l_sup + lambdas["kd"] * l_kd
    if l_con is not None:
        total = total + lambdas["con"] * l_con
    return total, {"sup": l_sup.item(), "kd": l_kd.item(),
                   "con": l_con.item() if l_con is not None else float("nan")}


# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, loader, device, lambdas, temperature, use_kd=True):
    """Returns mean loss components and per-label AUROC on validation."""
    model.eval()
    sums = {"total": 0.0, "sup": 0.0, "kd": 0.0, "con": 0.0}
    n = 0
    all_probs, all_targets, all_masks = [], [], []
    for images, targets, mask, teacher, c_score, c_map, c_valid in loader:
        images = images.to(device)
        targets, mask = targets.to(device), mask.to(device)
        teacher = teacher.to(device)
        c_score = c_score.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            d_logits, c_logits = model(images)
        # losses in fp32: BCE (not BCEWithLogits) is unsafe under autocast
        total, parts = total_loss(d_logits.float(), c_logits.float(), targets, mask,
                                  teacher, lambdas, temperature,
                                  concept_scores=c_score)
        bs = images.shape[0]
        sums["total"] += total.item() * bs
        sums["sup"] += parts["sup"] * bs
        sums["kd"] += parts["kd"] * bs
        sums["con"] += (parts["con"] if not math.isnan(parts["con"]) else 0.0) * bs
        n += bs
        all_probs.append(d_logits.float().sigmoid().cpu())
        all_targets.append(targets.cpu())
        all_masks.append(mask.cpu())

    probs = torch.cat(all_probs).numpy()
    tgts = torch.cat(all_targets).numpy()
    msks = torch.cat(all_masks).numpy().astype(bool)

    aurocs = []
    for i, name in enumerate(TASKS):
        y, p, m = tgts[:, i], probs[:, i], msks[:, i]
        y, p = y[m], p[m]
        if len(np.unique(y)) < 2:
            print(f"  WARNING: {name} has a single class in validation "
                  f"({int(y.sum())} positives); skipping AUROC")
            aurocs.append(float("nan"))
        else:
            aurocs.append(roc_auc_score(y, p))
    valid_auroc = [a for a in aurocs if not math.isnan(a)]
    macro = float(np.mean(valid_auroc)) if valid_auroc else float("nan")
    return {k: v / max(n, 1) for k, v in sums.items()}, aurocs, macro


# ---------------- Training ----------------
def train_epoch(model, loader, optimizer, scaler, device, lambdas, temperature):
    model.train()
    sums = {"total": 0.0, "sup": 0.0, "kd": 0.0, "con": 0.0}
    n = 0
    for images, targets, mask, teacher, c_score, c_map, c_valid in loader:
        images = images.to(device, non_blocking=True)
        targets, mask = targets.to(device), mask.to(device)
        teacher = teacher.to(device)
        c_score = c_score.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            d_logits, c_logits = model(images)
        total, parts = total_loss(d_logits.float(), c_logits.float(), targets, mask,
                                  teacher, lambdas, temperature,
                                  concept_scores=c_score)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.shape[0]
        sums["total"] += total.item() * bs
        sums["sup"] += parts["sup"] * bs
        sums["kd"] += parts["kd"] * bs
        sums["con"] += (parts["con"] if not math.isnan(parts["con"]) else 0.0) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def make_loaders(train_df, val_df, args, device, concepts=None,
                 train_perm=None, augment_train=True):
    train_aug = augment if augment_train else None
    train_ds = CheXpertDataset(train_df, _teacher_logits, concepts,
                               concept_perm=train_perm,
                               num_concepts=args.num_concepts,
                               augment_fn=train_aug, seed=args.seed)
    val_ds = CheXpertDataset(val_df, _teacher_logits, concepts,
                             num_concepts=args.num_concepts,
                             augment_fn=None, seed=args.seed)
    common = dict(num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


_teacher_logits = None  # set in main before make_loaders
_concepts = None        # set in main before make_loaders


def main():
    global _teacher_logits, _concepts
    args = parse_args()

    # ---- seed / device ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    if args.overfit:
        args.epochs = 30
        print(">>> OVERFIT MODE: train == eval on all frontal images, no augmentation")

    # ---- load labels (frontal only) ----
    df = load_labels(args.csv)
    print(f"Frontal images in {args.csv.name}: {len(df)}")

    # ---- teacher logits: assert coverage before training ----
    _teacher_logits = load_teacher_logits(args.logits)
    missing_t = [p for p in df["Path"] if _safe_teacher(p) is None]
    readable, unreadable = 0, []
    for p in df["Path"]:
        fp = fix_path(p)
        if fp.is_file() and cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) is not None:
            readable += 1
        else:
            unreadable.append(str(fp))
    print(f"Teacher logits: {len(df) - len(missing_t)}/{len(df)} images covered")
    print(f"Readable images: {readable}/{len(df)}")
    if missing_t:
        raise AssertionError(f"{len(missing_t)} images lack teacher logits; first: {missing_t[0]}")
    if unreadable:
        raise AssertionError(f"{len(unreadable)} images unreadable; first: {unreadable[0]}")

    # ---- concept targets (BioViL-T cache) ----
    uses_concepts = args.mode in {"B3", "Ours", "C1"}
    _concepts = None
    if args.concepts.is_file():
        _concepts = load_concepts(args.concepts)
        args.num_concepts = len(_concepts["names"])
        missing_c = [p for p in df["Path"] if _safe_concept(p) is None]
        print(f"Concepts: {args.num_concepts} concepts | coverage "
              f"{len(df) - len(missing_c)}/{len(df)} images")
        if missing_c:
            raise AssertionError(f"{len(missing_c)} images lack concept targets; "
                                 f"first: {missing_c[0]}")
    elif uses_concepts:
        raise FileNotFoundError(
            f"Mode {args.mode} needs concept targets but {args.concepts} is missing. "
            "Run cache_concpets.py first (or pass --concepts).")
    else:
        print(f"No concept file at {args.concepts}; modes {args.mode} do not need it")

    # ---- split ----
    if args.overfit:
        train_df = val_df = df
    else:
        train_df, val_df = patient_split(df, seed=args.seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} (patient-disjoint: "
          f"{not (set(train_df.patient_id) & set(val_df.patient_id))})")

    # ---- model ----
    model = Student(num_concepts=args.num_concepts).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dis = sum(p.numel() for p in model.disease_head.parameters())
    n_con = sum(p.numel() for p in model.concept_head.parameters())
    print(f"Params | encoder+disease head: {n_enc + n_dis:,} | concept head: {n_con:,}")

    lambdas = {"sup": args.lambda_sup, "kd": args.lambda_kd, "con": args.lambda_con}
    uses_kd = args.mode in {"B2", "Ours", "C1"}
    if not uses_kd:
        lambdas["kd"] = 0.0
    if not uses_concepts:
        lambdas["con"] = 0.0
    use_kd = uses_kd
    print(f"Mode {args.mode} | lambdas: sup={lambdas['sup']} kd={lambdas['kd']} "
          f"con={lambdas['con']}")

    # C1: permute concept (score, map) pairs across training images once, fixed seed
    train_perm = None
    if args.mode == "C1":
        train_perm = np.random.default_rng(args.seed).permutation(len(train_df))
        print("C1: concept targets shuffled across training images")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- loaders (batch-size fallback on OOM handled in train loop) ----
    bs = args.batch_size
    while True:
        try:
            train_loader, val_loader = make_loaders(train_df, val_df, args, device,
                                                    concepts=_concepts,
                                                    train_perm=train_perm,
                                                    augment_train=not args.overfit)
            # smoke-test one batch through the model to catch OOM early
            imgs, *_ = next(iter(train_loader))
            with torch.no_grad():
                model(imgs.to(device))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 8:
                raise
            bs //= 2
            args.batch_size = bs
            print(f"OOM -> reducing batch size to {bs}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- training loop ----
    best_macro, best_epoch = -1.0, -1
    history = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scaler, device,
                         lambdas, args.temperature)
        va, aurocs, macro = evaluate(model, val_loader, device,
                                     lambdas, args.temperature, use_kd=use_kd)
        scheduler.step()
        peak_mb = (torch.cuda.max_memory_allocated() / 2**20
                   if device.type == "cuda" else 0.0)
        rec = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()},
               "val_macro_auroc": macro, "peak_gpu_mb": peak_mb,
               "sec": time.time() - t0}
        history.append(rec)
        auc_str = " ".join(f"{n[:4]}={a:.3f}" if not math.isnan(a) else f"{n[:4]}=n/a"
                           for n, a in zip(TASKS, aurocs))
        print(f"Ep {epoch:02d}/{args.epochs} | "
              f"tr_loss {tr['total']:.4f} (sup {tr['sup']:.4f} kd {tr['kd']:.4f} "
              f"con {tr['con']:.4f}) | "
              f"val_loss {va['total']:.4f} | macroAUROC {macro:.4f} | "
              f"{auc_str} | peak {peak_mb:.0f}MB | {rec['sec']:.1f}s", flush=True)

        if not math.isnan(macro) and macro > best_macro:
            best_macro, best_epoch = macro, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_auroc": macro, "mode": args.mode,
                        "tasks": TASKS, "num_concepts": args.num_concepts,
                        "args": vars(args)},
                       args.output_dir / "student_best.pth")

    torch.save({"state_dict": model.state_dict(), "epoch": args.epochs,
                "val_macro_auroc": best_macro, "mode": args.mode,
                "tasks": TASKS, "num_concepts": args.num_concepts,
                "args": vars(args)},
               args.output_dir / "student_last.pth")

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    print(f"\nBest macro AUROC {best_macro:.4f} @ epoch {best_epoch}")
    print("Saved:", args.output_dir / "student_best.pth", "and student_last.pth")


def _safe_teacher(csv_path_str: str):
    try:
        return get_teacher(_teacher_logits, csv_path_str)
    except KeyError:
        return None


def _safe_concept(csv_path_str: str):
    try:
        return get_concept(_concepts, csv_path_str)
    except (KeyError, TypeError):
        return None


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
