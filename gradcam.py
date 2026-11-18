"""Standalone Grad-CAM / concept-CAM visualization for a saved checkpoint.

Does NOT train -- loads runs/<run>/best.pth (or --ckpt), rebuilds the student,
and writes overlays next to the checkpoint (or --out).

Examples (from Project-detail/Scripts):
    python gradcam.py                                    # newest best.pth
    python gradcam.py --ckpt runs/Ours+map_.../best.pth --n 3
    python gradcam.py --ckpt runs/<run>/last.pth --out runs/<run>/viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train as T
from common import (RUNS_DIR, TASKS, canonical_key, load_labels,
                    patient_split)
from students import build_student


def newest_ckpt() -> Path:
    cands = sorted(RUNS_DIR.glob("*/best.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("No runs/*/best.pth found; pass --ckpt")
    return cands[-1]


def main():
    p = argparse.ArgumentParser(description="Grad-CAM for a saved student ckpt")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--n", type=int, default=3,
                   help="val images per label (disease CAM) / images (concept CAM)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ckpt_path = (args.ckpt or newest_ckpt()).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = argparse.Namespace(**ckpt["args"])
    for k in ("csv", "teacher_cache", "concept_cache", "val_csv"):
        v = getattr(cargs, k, None)
        if v is not None:
            setattr(cargs, k, Path(v))
    print("Checkpoint:", ckpt_path)
    print("Run config: mode", cargs.mode, "| student", cargs.student,
          "| epoch", ckpt.get("epoch"), "| macroAUROC", ckpt.get("val_macro_auroc"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student(cargs.student, num_concepts=ckpt["num_concepts"],
                          pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- rebuild the same validation split the run used ----
    df, targets, mask = load_labels(
        cargs.csv, blank_policy=cargs.blank_policy,
        uncertain_policy=cargs.uncertain_policy)
    tblob = T._load_teacher_cache(cargs.teacher_cache, df)
    cblob = T._load_concept_cache(cargs.concept_cache, df, cargs.mode)

    if getattr(cargs, "overfit", False):
        val_df, val_targets, val_mask = df, targets, mask
    elif getattr(cargs, "val_csv", None) is not None:
        vdf, vtargets, vmask = load_labels(
            cargs.val_csv, blank_policy=cargs.blank_policy,
            uncertain_policy=cargs.uncertain_policy)
        val_df, val_targets, val_mask = vdf, vtargets, vmask
    else:
        _, val_df = patient_split(df, val_frac=cargs.val_frac, seed=cargs.seed)
        k2i = {canonical_key(p): i for i, p in enumerate(df["Path"])}
        va_idx = [k2i[canonical_key(p)] for p in val_df["Path"]]
        val_targets, val_mask = targets[va_idx], mask[va_idx]

    val_ds = T.CheXpertDataset(val_df, val_targets, val_mask,
                               teacher_logits=tblob, concept_blob=cblob,
                               seed=cargs.seed)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    probs, tgts, msks, _, _ = T.predict_val(model, loader, device)

    out_dir = (args.out or ckpt_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    T.gradcam_overlays(model, val_ds, val_df, probs, tgts, msks, out_dir, args.n)
    if cblob is not None:
        T.concept_cam_figures(model, val_ds, cblob["concepts"], out_dir,
                              n_images=args.n)
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError,
            AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
