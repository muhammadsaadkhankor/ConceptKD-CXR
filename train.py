"""Single training/evaluation entry point for the concept-guided KD project.

Teacher, student and concept source are all swappable: teacher/student via
--teacher-cache/--student (or configs/default.yaml), concept targets via
--concept-cache. Neither the teacher nor BioViL-T is ever loaded here -- only
their cached logits/maps.

Examples (from Project-detail/Scripts):
    python train.py --mode Ours --epochs 30
    python train.py --mode Ours+map --overfit --epochs 30
    python train.py --mode C1 --epochs 2
    python train.py --mode Ours --student resnet18 --epochs 2
    python train.py --eval-only --ckpt runs/Ours_effnet_b0_xray-bench_convnext_tiny_biovilt_s42/best.pth
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import (BASE, CACHE_DIR, DATA_DIR, INPUT_H, INPUT_W, RUNS_DIR,
                    TASKS, NUM_CLASSES, aspect_fit, augment, canonical_key,
                    load_labels, patient_split, resolve_image, to_tensor)
from students import STUDENTS, build_student, param_report

SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPTS_DIR / "configs" / "default.yaml"

MODES = ["B1", "B2", "B3", "Ours", "C1", "C2_placeholder", "Ours+map"]
NEEDS_TEACHER = {"B2", "Ours", "C1", "C2_placeholder", "Ours+map"}
NEEDS_CONCEPTS = {"B3", "Ours", "C1", "C2_placeholder", "Ours+map"}
USES_KD = NEEDS_TEACHER
USES_CON = NEEDS_CONCEPTS
USES_MAP = {"Ours+map"}


# ---------------- Config ----------------
def parse_args():
    cfg = {}
    if CONFIG_PATH.is_file():
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
    p = argparse.ArgumentParser(description="CheXpert concept-guided KD trainer")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--teacher-cache", type=Path, default=None)
    p.add_argument("--concept-cache", type=Path, default=None)
    p.add_argument("--student", default=None, choices=sorted(STUDENTS))
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--teacher", default=None, help="teacher spec for run-name/cache hint")
    p.add_argument("--concept", default=None, help="concept source for run-name")
    p.add_argument("--lambda-sup", type=float, default=None)
    p.add_argument("--lambda-kd", type=float, default=None)
    p.add_argument("--lambda-con", type=float, default=None)
    p.add_argument("--lambda-map", type=float, default=None)
    p.add_argument("--kd-temp", type=float, default=None)
    p.add_argument("--map-temp", type=float, default=None)
    p.add_argument("--blank-policy", default=None, choices=["zero", "mask"])
    p.add_argument("--uncertain-policy", default=None, choices=["mask", "ones", "zeros"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=None)
    p.add_argument("--val-csv", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-concepts", type=int, default=None)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--gradcam", type=int, default=None,
                   help="save Grad-CAM overlays for N val images per label")
    cli = p.parse_args()

    # merge: start from CLI namespace (has all argparse defaults), then fill
    # any still-None value from yaml. CLI always wins.
    args = argparse.Namespace(**vars(cli))
    for k, v in cfg.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if getattr(args, "gradcam", None) is None:
        args.gradcam = 0
    if getattr(args, "num_workers", None) is None:
        args.num_workers = 0
    # normalize paths relative to Scripts/
    for k in ("csv", "teacher_cache", "concept_cache", "val_csv", "ckpt"):
        v = getattr(args, k, None)
        if v is not None and not Path(v).is_absolute():
            setattr(args, k, (SCRIPTS_DIR / v).resolve())
    if args.csv is None:
        args.csv = DATA_DIR / "train_100.csv"
    if args.teacher_cache is None:
        args.teacher_cache = CACHE_DIR / "teacher_xray-bench_convnext_tiny.pt"
    if args.concept_cache is None:
        args.concept_cache = CACHE_DIR / "concepts_biovilt.pt"
    return args


def default_run_name(args):
    teacher = args.teacher or _cache_teacher_name(args.teacher_cache)
    concept = args.concept or _cache_source_name(args.concept_cache)
    student = args.student or "effnet_b0"
    safe = lambda s: str(s).replace(":", "-").replace("/", "-")
    return f"{args.mode}_{safe(student)}_{safe(teacher)}_{safe(concept)}_s{args.seed}"


def _cache_teacher_name(path):
    try:
        b = torch.load(path, map_location="cpu", weights_only=False)
        return b.get("teacher", Path(path).stem.replace("teacher_", ""))
    except Exception:
        return Path(path).stem.replace("teacher_", "")


def _cache_source_name(path):
    try:
        b = torch.load(path, map_location="cpu", weights_only=False)
        return b.get("source", Path(path).stem.replace("concepts_", ""))
    except Exception:
        return Path(path).stem.replace("concepts_", "")


# ---------------- Data ----------------
class CheXpertDataset(Dataset):
    """Returns (image, targets, mask, teacher[5], c_score[C], c_map[C,40,48],
    map_valid[40,48]). Missing caches -> zero tensors (losses are lambda-0)."""

    def __init__(self, df, targets, mask, teacher_logits=None, concept_blob=None,
                 concept_perm=None, augment_fn=None, seed=42):
        self.df = df.reset_index(drop=True)
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))
        self.augment_fn = augment_fn
        self.rng = np.random.default_rng(seed)
        self.paths = [resolve_image(canonical_key(p)) for p in self.df["Path"]]
        self.keys = [canonical_key(p) for p in self.df["Path"]]
        self.concept_perm = concept_perm
        C = 10

        if teacher_logits is not None:
            self.teacher = torch.stack([teacher_logits[k].float() for k in self.keys])
        else:
            self.teacher = torch.zeros(len(self.df), NUM_CLASSES)

        if concept_blob is not None:
            C = len(concept_blob["concepts"])
            self.c_score = torch.stack([concept_blob["scores"][k].float()
                                        for k in self.keys])
            self.c_map = torch.stack([concept_blob["maps"][k].float()
                                      for k in self.keys])
            self.c_valid = torch.stack([concept_blob["map_valid"][k].float()
                                        for k in self.keys])
        else:
            self.c_score = torch.zeros(len(self.df), C)
            gh, gw = (concept_blob or {}).get("grid_hw", (40, 48)) \
                if concept_blob else (40, 48)
            self.c_map = torch.zeros(len(self.df), C, gh, gw)
            self.c_valid = torch.ones(len(self.df), gh, gw)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.paths[idx]}")
        img = aspect_fit(img, INPUT_H, INPUT_W)
        if self.augment_fn is not None:
            img = augment_fn(img, self.rng)
        cidx = idx if self.concept_perm is None else int(self.concept_perm[idx])
        return (to_tensor(img), self.targets[idx], self.mask[idx],
                self.teacher[idx], self.c_score[cidx], self.c_map[cidx],
                self.c_valid[cidx])


# ---------------- Losses (all fp32, outside autocast) ----------------
def loss_sup(disease_logits, targets, mask):
    per = F.binary_cross_entropy_with_logits(disease_logits, targets,
                                             reduction="none")
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def loss_kd(student_logits, teacher_logits, temperature):
    s = torch.sigmoid(student_logits / temperature)
    t = torch.sigmoid(teacher_logits / temperature)
    return F.binary_cross_entropy(s, t, reduction="mean") * temperature ** 2


def loss_con(concept_logits, concept_scores):
    return F.binary_cross_entropy(torch.sigmoid(concept_logits),
                                  concept_scores, reduction="mean")


def loss_map(concept_cam, c_map, c_valid, c_score, map_temp):
    """KL(softmax(target_map/T) || softmax(student_cam)) over valid cells,
    for each (image, concept) with cached score > 0.5. Averaged over pairs."""
    B, C, gh, gw = concept_cam.shape
    total, count = 0.0, 0
    for b in range(B):
        for k in range(C):
            if c_score[b, k].item() <= 0.5:
                continue
            tgt = c_map[b, k]                       # [40,48] cached grid
            val = c_valid[b]                        # [40,48]
            if val.sum() < 4:
                continue
            # resize cached map+valid to student grid
            t = tgt[None, None].float()
            v = val[None, None].float()
            t = F.adaptive_avg_pool2d(t, (gh, gw))[0, 0]
            v = F.adaptive_avg_pool2d(v, (gh, gw))[0, 0] > 0.5
            if v.sum() < 4:
                continue
            p = F.softmax((t / map_temp)[v], dim=0)
            q = F.softmax(concept_cam[b, k].float()[v], dim=0)
            total = total + F.kl_div(q.log().clamp_min(-20), p,
                                     reduction="sum", log_target=False)
            count += 1
    if count == 0:
        return concept_cam.sum() * 0.0
    return total / count


def compute_losses(out, batch, lambdas, kd_temp, map_temp):
    (_, targets, mask, teacher, c_score, c_map, c_valid) = batch
    d_logits = out["disease_logits"].float()
    c_logits = out["concept_logits"].float()
    l_sup = loss_sup(d_logits, targets, mask)
    l_kd = loss_kd(d_logits, teacher, kd_temp) if lambdas["kd"] > 0 else d_logits.sum() * 0
    l_con = loss_con(c_logits, c_score) if lambdas["con"] > 0 else c_logits.sum() * 0
    l_map = (loss_map(out["concept_cam"].float(), c_map, c_valid, c_score,
                      map_temp) if lambdas["map"] > 0 else c_logits.sum() * 0)
    total = (lambdas["sup"] * l_sup + lambdas["kd"] * l_kd +
             lambdas["con"] * l_con + lambdas["map"] * l_map)
    return total, {"sup": l_sup.item(), "kd": l_kd.item(),
                   "con": l_con.item(), "map": l_map.item()}


# ---------------- Evaluation ----------------
@torch.no_grad()
def predict_val(model, loader, device):
    model.eval()
    probs, tgts, msks, cprobs, cscores = [], [], [], [], []
    for images, targets, mask, teacher, c_score, c_map, c_valid in loader:
        images = images.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(images)
        probs.append(out["disease_logits"].float().sigmoid().cpu())
        tgts.append(targets)
        msks.append(mask)
        cprobs.append(out["concept_logits"].float().sigmoid().cpu())
        cscores.append(c_score)
    return (torch.cat(probs).numpy(), torch.cat(tgts).numpy(),
            torch.cat(msks).numpy(), torch.cat(cprobs).numpy(),
            torch.cat(cscores).numpy())


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def val_metrics(probs, tgts, msks):
    """Per-label AUROC/macro + masked arrays for later threshold metrics."""
    aurocs = []
    for i, name in enumerate(TASKS):
        y, p, m = tgts[:, i], probs[:, i], msks[:, i].astype(bool)
        y, p = y[m], p[m]
        if len(np.unique(y)) < 2:
            print(f"  WARNING: {name} single class in val "
                  f"({int(y.sum())} pos); skipping AUROC")
            aurocs.append(float("nan"))
        else:
            aurocs.append(_safe_auc(y, p))
    valid = [a for a in aurocs if not math.isnan(a)]
    return aurocs, float(np.mean(valid)) if valid else float("nan")


def concept_val_metrics(cprobs, cscores):
    """Concept AUROC vs binarized (>0.5) scores + MAE. Returns (macro_auc, mae)."""
    aucs, maes = [], []
    for k in range(cprobs.shape[1]):
        y = (cscores[:, k] > 0.5).astype(int)
        p = cprobs[:, k]
        if len(np.unique(y)) == 2:
            aucs.append(_safe_auc(y, p))
        maes.append(float(np.mean(np.abs(p - cscores[:, k]))))
    return (float(np.mean(aucs)) if aucs else float("nan"), float(np.mean(maes)))


def final_disease_metrics(probs, tgts, msks):
    """AUROC/AUPRC/macro, Youden-J F1/sens/spec, Brier, ECE(15)+reliability."""
    out = {"per_label_auroc": {}, "per_label_auprc": {},
           "per_label_f1": {}, "per_label_sensitivity": {},
           "per_label_specificity": {}, "per_label_threshold": {},
           "per_label_brier": {}}
    all_y, all_p = [], []
    for i, name in enumerate(TASKS):
        y, p, m = tgts[:, i], probs[:, i], msks[:, i].astype(bool)
        y, p = y[m], p[m]
        out["per_label_auroc"][name] = _safe_auc(y, p)
        out["per_label_auprc"][name] = (
            average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan"))
        out["per_label_brier"][name] = float(brier_score_loss(y, p))
        # Youden-J threshold on validation
        best_j, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            pred = p >= t
            tp = ((pred == 1) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1
            if j > best_j:
                best_j, best_t = j, float(t)
        pred = (p >= best_t).astype(int)
        f1, sens, spec, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        out["per_label_threshold"][name] = best_t
        out["per_label_f1"][name] = float(f1)
        out["per_label_sensitivity"][name] = float(sens)
        out["per_label_specificity"][name] = float(spec)
        all_y.append(y)
        all_p.append(p)

    def macro(key):
        vals = [v for v in out[key].values() if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out["macro_auroc"] = macro("per_label_auroc")
    out["macro_auprc"] = macro("per_label_auprc")
    out["macro_f1"] = macro("per_label_f1")

    # ECE with 15 bins over all masked (label,image) pairs
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    ece, bins = _ece(y, p, n_bins=15)
    out["ece"] = ece
    out["ece_bins"] = bins
    return out


def _ece(y, p, n_bins=15):
    bins = []
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                         "acc": None, "conf": None, "n": 0})
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        ece += (m.sum() / len(p)) * abs(acc - conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "acc": acc, "conf": conf, "n": int(m.sum())})
    return float(ece), bins


def plot_reliability(bins, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"] > 0]
    acc = [b["acc"] for b in bins if b["n"] > 0]
    conf = [b["conf"] for b in bins if b["n"] > 0]
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.plot(conf, acc, "o-")
    plt.xlabel("confidence"); plt.ylabel("accuracy")
    plt.title("Reliability diagram (15 bins)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def compute_macs(model, device):
    x = torch.zeros(1, 3, INPUT_H, INPUT_W)
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return int(macs)
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        return int(FlopCountAnalysis(model, x).total()) // 2  # flops->MACs approx
    except Exception:
        return None


def compute_latency(model, device):
    model.eval()
    x = torch.zeros(1, 3, INPUT_H, INPUT_W).to(device)
    fp16 = device.type == "cuda"
    with torch.no_grad():
        for _ in range(20):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / 100 * 1000  # ms/image


# ---------------- Grad-CAM ----------------
def gradcam_overlays(model, val_ds, val_df, probs, tgts, msks, run_dir, n_per_label):
    """Save Grad-CAM overlays for up to n_per_label val images per label."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return
    out_dir = run_dir / "gradcam"
    out_dir.mkdir(exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    # pick indices: for each label, images where that label is positive
    chosen = []
    for i, name in enumerate(TASKS):
        pos_idx = np.where((tgts[:, i] == 1) & (msks[:, i] > 0))[0]
        for j in pos_idx[:n_per_label]:
            chosen.append((name, int(j)))
    if not chosen:
        return

    import matplotlib.pyplot as plt
    for label, idx in chosen:
        item = val_ds[idx]
        img_t = item[0][None].to(device)
        feat = model.encoder.forward_features(img_t)
        feat.retain_grad()
        pooled = feat.mean(dim=(2, 3))
        li = TASKS.index(label)
        logit = model.disease_head(pooled)[0, li]
        model.zero_grad(set_to_none=True)
        logit.backward()
        w = feat.grad.mean(dim=(2, 3))[0]             # [C]
        cam = torch.relu(torch.einsum("chw,c->hw", feat[0], w))
        cam = cam - cam.min()
        cam = cam / cam.clamp_min(1e-8)
        cam_np = cam.detach().cpu().numpy()

        raw = cv2.imread(str(val_ds.paths[idx]), cv2.IMREAD_GRAYSCALE)
        base = aspect_fit(raw, INPUT_H, INPUT_W)
        base_rgb = np.repeat(base[:, :, None], 3, axis=2) / 255.0
        gh, gw = feat.shape[-2:]
        cam_up = cv2.resize((cam_np * 255).astype(np.uint8),
                              (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        heat_rgb = cv2.applyColorMap(cam_up, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
        overlay = base_rgb * 0.6 + heat_rgb * 0.4
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1); plt.imshow(base, cmap="gray"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(np.clip(overlay, 0, 1)); plt.axis("off")
        plt.title(f"{label} (p={probs[idx, li]:.2f})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label}_{idx}.png", dpi=100)
        plt.close()
    print(f"  Grad-CAM overlays -> {out_dir}")


def _norm01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / max(a.max() - a.min(), 1e-8)


def _jet_overlay(base_rgb, heat01):
    """base_rgb [H,W,3] 0-1 + heat01 [H,W] 0-1 -> JET-colored overlay."""
    heat_up = cv2.resize((np.clip(heat01, 0, 1) * 255).astype(np.uint8),
                         (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    heat_rgb = cv2.applyColorMap(heat_up, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    return base_rgb * 0.6 + heat_rgb * 0.4


def concept_cam_figures(model, val_ds, concept_names, run_dir, n_images=3):
    """For a few val images: student concept CAM vs cached BioViL-T map.

    Rows = concepts; cols = [image, student CAM overlay, cached map overlay].
    Cached-map padding stays dark via map_valid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir = run_dir / "concept_cam"
    out_dir.mkdir(exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    C = len(concept_names)
    for idx in range(min(n_images, len(val_ds))):
        item = val_ds[idx]
        img_t = item[0][None].to(device)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(img_t)
        cam = out["concept_cam"][0].float().cpu().numpy()     # [C,h,w]
        c_map = item[5].numpy()                              # [C,40,48]
        c_valid = item[6].numpy()                            # [40,48]
        c_score = item[4].numpy()                            # [C]

        raw = cv2.imread(str(val_ds.paths[idx]), cv2.IMREAD_GRAYSCALE)
        base = aspect_fit(raw, INPUT_H, INPUT_W)
        base_rgb = np.repeat(base[:, :, None], 3, axis=2) / 255.0
        valid_up = (cv2.resize(c_valid, (INPUT_W, INPUT_H),
                               interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)

        fig, axes = plt.subplots(C, 3, figsize=(9, 1.8 * C))
        for k in range(C):
            stu = _jet_overlay(base_rgb, _norm01(cam[k]))
            tgt = _jet_overlay(base_rgb, _norm01(c_map[k]) * valid_up)
            axes[k, 0].imshow(base, cmap="gray"); axes[k, 0].axis("off")
            axes[k, 1].imshow(np.clip(stu, 0, 1)); axes[k, 1].axis("off")
            axes[k, 2].imshow(np.clip(tgt, 0, 1)); axes[k, 2].axis("off")
            axes[k, 0].set_ylabel(f"{concept_names[k]}\ns={c_score[k]:.2f}",
                                  fontsize=7)
        axes[0, 0].set_title("image", fontsize=8)
        axes[0, 1].set_title("student CAM", fontsize=8)
        axes[0, 2].set_title("cached map", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"concepts_img{idx}.png", dpi=110)
        plt.close()
    print(f"  Concept CAM figures -> {out_dir}")


# ---------------- Main ----------------
def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = default_run_name(args)
    run_dir = RUNS_DIR / args.run_name

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Run:", run_dir)

    # ---- eval-only path ----
    if args.eval_only:
        if args.ckpt is None:
            raise ValueError("--eval-only requires --ckpt")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        run_dir = args.ckpt.resolve().parent       # write metrics next to ckpt
        cargs = ckpt.get("args", {})
        cargs = argparse.Namespace(**{k: v for k, v in cargs.items()})
        model = build_student(cargs.student, num_concepts=cargs.num_concepts,
                              pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        df, targets, mask = load_labels(
            Path(cargs.csv), blank_policy=cargs.blank_policy,
            uncertain_policy=cargs.uncertain_policy)
        tblob = _load_teacher_cache(cargs.teacher_cache, df)
        cblob = _load_concept_cache(cargs.concept_cache, df, cargs.mode)
        _assert_coverage(df, tblob, cblob)
        if getattr(cargs, "overfit", False):
            val_df = df
            val_targets, val_mask = targets, mask
        elif getattr(cargs, "val_csv", None):
            vdf, vtargets, vmask = load_labels(
                Path(cargs.val_csv), blank_policy=cargs.blank_policy,
                uncertain_policy=cargs.uncertain_policy)
            val_df, val_targets, val_mask = vdf, vtargets, vmask
        else:
            _, val_df = patient_split(df, val_frac=cargs.val_frac,
                                      seed=cargs.seed)
            k2i = {canonical_key(p): i for i, p in enumerate(df["Path"])}
            va_idx = [k2i[canonical_key(p)] for p in val_df["Path"]]
            val_targets, val_mask = targets[va_idx], mask[va_idx]
        probs, tgts, msks, cprobs, cscores = _run_eval_data(
            model, val_df, val_targets, val_mask, df, tblob, cblob,
            cargs, device)
        metrics = _assemble_metrics(model, probs, tgts, msks, cprobs, cscores,
                                    device, run_dir, cargs)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"metrics.json -> {run_dir / 'metrics.json'}")
        return

    # ---- mode/lambda resolution ----
    lambdas = {"sup": args.lambda_sup, "kd": args.lambda_kd,
               "con": args.lambda_con, "map": args.lambda_map}
    if args.mode == "C2_placeholder":
        print("NOTE: C2_placeholder behaves like Ours (placeholder mode).")
    if args.mode not in NEEDS_TEACHER:
        lambdas["kd"] = 0.0
    if args.mode not in NEEDS_CONCEPTS:
        lambdas["con"] = 0.0
    if args.mode not in USES_MAP:
        lambdas["map"] = 0.0
    print(f"Mode {args.mode} | lambdas {lambdas} | kd_temp={args.kd_temp} "
          f"map_temp={args.map_temp}")

    # ---- data ----
    if args.overfit:
        args.epochs = 30
        print(">>> OVERFIT: train==eval on all frontal images, no augmentation")
    df, targets_all, mask_all = load_labels(
        args.csv, blank_policy=args.blank_policy,
        uncertain_policy=args.uncertain_policy)
    print(f"Frontal images: {len(df)}")

    tblob = _load_teacher_cache(args.teacher_cache, df) if args.mode in NEEDS_TEACHER else None
    cblob = _load_concept_cache(args.concept_cache, df, args.mode) \
        if args.mode in NEEDS_CONCEPTS else None
    if args.mode in NEEDS_TEACHER and tblob is None:
        raise FileNotFoundError(
            f"Mode {args.mode} needs teacher cache {args.teacher_cache} "
            "(run: python teachers.py cache ...)")
    if args.mode in NEEDS_CONCEPTS and cblob is None:
        raise FileNotFoundError(
            f"Mode {args.mode} needs concept cache {args.concept_cache} "
            "(run: python concepts.py cache ...)")

    # assert coverage + readability
    _assert_coverage(df, tblob, cblob)

    # ---- split ----
    if args.val_csv is not None:
        vdf, vtargets, vmask = load_labels(
            args.val_csv, blank_policy=args.blank_policy,
            uncertain_policy=args.uncertain_policy)
        train_df, val_df = df, vdf
    elif args.overfit:
        train_df = val_df = df
    else:
        train_df, val_df = patient_split(df, val_frac=args.val_frac,
                                       seed=args.seed)

    # index-aligned targets/mask for each frame (df row order == targets_all)
    k2i = {canonical_key(p): i for i, p in enumerate(df["Path"])}
    tr_idx = [k2i[canonical_key(p)] for p in train_df["Path"]]
    train_targets = targets_all[tr_idx]
    train_mask = mask_all[tr_idx]
    if args.val_csv is not None:
        val_targets, val_mask = vtargets, vmask      # labels from the val CSV
    else:
        va_idx = [k2i[canonical_key(p)] for p in val_df["Path"]]
        val_targets = targets_all[va_idx]
        val_mask = mask_all[va_idx]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # C1: permute concept (score,map,valid) across TRAIN images once, fixed seed
    train_perm = None
    if args.mode == "C1":
        train_perm = np.random.default_rng(args.seed).permutation(len(train_df))
        print("C1: concept targets permuted across training images")

    # ---- model ----
    num_concepts = args.num_concepts
    if cblob is not None:
        num_concepts = len(cblob["concepts"])
    model = build_student(args.student, num_classes=NUM_CLASSES,
                          num_concepts=num_concepts, pretrained=True).to(device)
    n_inf, n_con = param_report(model)
    print(f"Params | inference path: {n_inf:,} | concept head: {n_con:,}")

    # C1 sanity print: first train image scores under C1 vs Ours
    if args.mode == "C1" and cblob is not None:
        k0 = canonical_key(train_df["Path"].iloc[0])
        true_scores = cblob["scores"][k0].numpy()
        perm_scores = cblob["scores"][
            canonical_key(train_df["Path"].iloc[int(train_perm[0])])].numpy()
        print("C1 check | image0 true concept scores:",
              np.round(true_scores, 3))
        print("C1 check | image0 permuted scores:  ",
              np.round(perm_scores, 3), "| differ:",
              not np.allclose(true_scores, perm_scores))

    # ---- loaders with OOM batch fallback ----
    common = dict(num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"))
    bs = args.batch_size
    while True:
        try:
            train_ds = CheXpertDataset(train_df, train_targets, train_mask,
                                       teacher_logits=tblob, concept_blob=cblob,
                                       concept_perm=train_perm,
                                       augment_fn=(None if args.overfit else augment),
                                       seed=args.seed)
            val_ds = CheXpertDataset(val_df, val_targets, val_mask,
                                     teacher_logits=tblob, concept_blob=cblob,
                                     seed=args.seed)
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **common)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **common)
            imgs = next(iter(train_loader))[0]
            with torch.no_grad():
                model(imgs.to(device))
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 8:
                raise
            bs //= 2
            print(f"OOM -> batch size {bs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    run_dir.mkdir(parents=True, exist_ok=True)
    # persist config
    cfg_used = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()}
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg_used))

    # ---- training loop ----
    best_macro, best_epoch = -1.0, -1
    history = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scaler, device,
                         lambdas, args.kd_temp, args.map_temp)
        va = evaluate_epoch(model, val_loader, device, lambdas,
                            args.kd_temp, args.map_temp, val_targets, val_mask)
        scheduler.step()
        peak_mb = (torch.cuda.max_memory_allocated() / 2**20
                   if device.type == "cuda" else 0.0)
        rec = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()},
               "peak_gpu_mb": peak_mb, "sec": time.time() - t0}
        history.append(rec)
        auc_str = " ".join(
            f"{n[:4]}={a:.3f}" if not math.isnan(a) else f"{n[:4]}=n/a"
            for n, a in zip(TASKS, va["aurocs"]))
        print(f"Ep {epoch:02d}/{args.epochs} | "
              f"tr {tr['total']:.4f} (sup {tr['sup']:.4f} kd {tr['kd']:.4f} "
              f"con {tr['con']:.4f} map {tr['map']:.4f}) | "
              f"val_loss {va['total']:.4f} | macroAUROC {va['macro']:.4f} | "
              f"conAUROC {va['con_auc']:.3f} conMAE {va['con_mae']:.3f} | "
              f"{auc_str} | peak {peak_mb:.0f}MB | {rec['sec']:.1f}s",
              flush=True)
        if not math.isnan(va["macro"]) and va["macro"] > best_macro:
            best_macro, best_epoch = va["macro"], epoch
            torch.save(_ckpt_dict(model, args, epoch, va["macro"], best=True),
                       run_dir / "best.pth")

    torch.save(_ckpt_dict(model, args, args.epochs, best_macro, best=False),
               run_dir / "last.pth")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # ---- final metrics on validation ----
    probs, tgts, msks, cprobs, cscores = predict_val(model, val_loader, device)
    metrics = _assemble_metrics(model, probs, tgts, msks, cprobs, cscores,
                                device, run_dir, args, history=history,
                                best_epoch=best_epoch)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.gradcam and args.gradcam > 0:
        gradcam_overlays(model, val_ds, val_df, probs, tgts, msks,
                         run_dir, args.gradcam)
        if cblob is not None:
            concept_cam_figures(model, val_ds, cblob["concepts"],
                                run_dir, n_images=args.gradcam)
    print(f"\nBest macro AUROC {best_macro:.4f} @ epoch {best_epoch}")
    print(f"Saved run -> {run_dir} (metrics.json, history.csv, best.pth, last.pth)")


def _ckpt_dict(model, args, epoch, macro, best):
    return {"state_dict": model.state_dict(), "epoch": epoch,
            "val_macro_auroc": macro, "mode": args.mode,
            "student": args.student, "num_concepts": args.num_concepts,
            "tasks": TASKS,
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "best": best}


def _load_teacher_cache(path, df):
    if path is None or not Path(path).is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    logits = {k: v.float() for k, v in blob["logits"].items()}
    return logits


def _load_concept_cache(path, df, mode):
    if path is None or not Path(path).is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "map_valid" not in blob:
        raise ValueError(f"{path} has no map_valid; re-run concepts.py cache")
    return blob


def _assert_coverage(df, tblob, cblob):
    keys = [canonical_key(p) for p in df["Path"]]
    readable = sum(1 for k in keys
                   if resolve_image(k).is_file()
                   and cv2.imread(str(resolve_image(k)),
                                  cv2.IMREAD_GRAYSCALE) is not None)
    print(f"Readable images: {readable}/{len(keys)}")
    if tblob is not None:
        missing = [k for k in keys if k not in tblob]
        print(f"Teacher coverage: {len(keys)-len(missing)}/{len(keys)}")
        if missing:
            raise AssertionError(f"{len(missing)} images lack teacher logits; "
                                 f"first: {missing[0]}")
    if cblob is not None:
        missing = [k for k in keys if k not in cblob["scores"]]
        print(f"Concept coverage: {len(keys)-len(missing)}/{len(keys)} "
              f"({len(cblob['concepts'])} concepts)")
        if missing:
            raise AssertionError(f"{len(missing)} images lack concept targets; "
                                 f"first: {missing[0]}")
    if readable != len(keys):
        raise AssertionError(f"{len(keys)-readable} images unreadable")


def _run_eval_data(model, val_df, val_targets, val_mask, df, tblob, cblob,
                   args, device):
    """Build val loader and return prediction arrays (eval-only path)."""
    val_ds = CheXpertDataset(val_df, val_targets, val_mask,
                             teacher_logits=tblob, concept_blob=cblob,
                             seed=args.seed)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    return predict_val(model, loader, device)


def _assemble_metrics(model, probs, tgts, msks, cprobs, cscores, device,
                      run_dir, args, history=None, best_epoch=None):
    dm = final_disease_metrics(probs, tgts, msks)
    con_auc, con_mae = concept_val_metrics(cprobs, cscores)
    pearson = float("nan")
    if cscores.size:
        ps, ys = [], []
        for k in range(cscores.shape[1]):
            if np.std(cscores[:, k]) > 0 and np.std(cprobs[:, k]) > 0:
                ps.append(cprobs[:, k]); ys.append(cscores[:, k])
        if ps:
            pearson = float(np.corrcoef(np.concatenate(ps),
                                        np.concatenate(ys))[0, 1])
    n_inf, n_con = param_report(model)
    metrics = {
        "mode": args.mode, "student": args.student, "run_name": args.run_name,
        "macro_auroc": dm["macro_auroc"], "macro_auprc": dm["macro_auprc"],
        "macro_f1": dm["macro_f1"], "ece": dm["ece"],
        "ece_bins": dm["ece_bins"],
        "per_label": {n: {"auroc": dm["per_label_auroc"][n],
                          "auprc": dm["per_label_auprc"][n],
                          "f1": dm["per_label_f1"][n],
                          "sensitivity": dm["per_label_sensitivity"][n],
                          "specificity": dm["per_label_specificity"][n],
                          "threshold_youden_j": dm["per_label_threshold"][n],
                          "brier": dm["per_label_brier"][n]}
                      for n in TASKS},
        "concept_macro_auroc": con_auc, "concept_mae": con_mae,
        "concept_pearson": pearson,
        "params_inference": n_inf, "params_concept_head": n_con,
        "macs": compute_macs(model, device),
        "latency_ms_per_image": compute_latency(model, device),
        "peak_gpu_mb": max((r.get("peak_gpu_mb", 0) for r in (history or [])),
                           default=0.0),
        "best_epoch": best_epoch,
        "n_val": int(msks.shape[0]),
    }
    plot_reliability(dm["ece_bins"], run_dir / "reliability.png")
    metrics["reliability_png"] = "reliability.png"
    return metrics


def train_epoch(model, loader, optimizer, scaler, device, lambdas,
                kd_temp, map_temp):
    model.train()
    sums = {"total": 0.0, "sup": 0.0, "kd": 0.0, "con": 0.0, "map": 0.0}
    n = 0
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        batch = tuple(t.to(device) if torch.is_tensor(t) else t
                      for t in batch[1:])
        batch = (images,) + batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(images)
        total, parts = compute_losses(out, batch, lambdas, kd_temp, map_temp)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.shape[0]
        for k in sums:
            sums[k] += (total.item() if k == "total" else parts[k]) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, device, lambdas, kd_temp, map_temp,
                   val_targets, val_mask):
    model.eval()
    sums = {"total": 0.0, "sup": 0.0, "kd": 0.0, "con": 0.0, "map": 0.0}
    n = 0
    probs, cprobs, cscores = [], [], []
    for batch in loader:
        images = batch[0].to(device)
        rest = tuple(t.to(device) for t in batch[1:])
        b = (images,) + rest
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model(images)
        total, parts = compute_losses(out, b, lambdas, kd_temp, map_temp)
        bs = images.shape[0]
        for k in sums:
            sums[k] += (total.item() if k == "total" else parts[k]) * bs
        n += bs
        probs.append(out["disease_logits"].float().sigmoid().cpu())
        cprobs.append(out["concept_logits"].float().sigmoid().cpu())
        cscores.append(b[4].cpu())
    probs = torch.cat(probs).numpy()
    cprobs = torch.cat(cprobs).numpy()
    cscores = torch.cat(cscores).numpy()
    aurocs, macro = val_metrics(probs, val_targets, val_mask)
    con_auc, con_mae = concept_val_metrics(cprobs, cscores)
    res = {k: v / max(n, 1) for k, v in sums.items()}
    res.update({"aurocs": aurocs, "macro": macro,
                "con_auc": con_auc, "con_mae": con_mae})
    return res


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError,
            AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
