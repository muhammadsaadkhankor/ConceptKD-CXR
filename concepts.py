"""Concept-source registry + CLI to cache BioViL-T concept targets.

Examples (from Project-detail/Scripts):
    python concepts.py selftest
    python concepts.py cache --source biovilt --csv ..\\Dataset\\train_100.csv
    python concepts.py viz --key train/patient41582/study5/view1_frontal.jpg

Cache files land in Scripts/cache/concepts_<source>.pt. Concept maps are stored
on a canonical 40x48 grid (stride 8 of the 320x384 student input) with a shared
map_valid mask. Training never loads BioViL-T -- only the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (CACHE_DIR, INPUT_H, INPUT_W, SCRIPTS_DIR, TASKS,
                    aspect_fit_float, canonical_key, load_labels, resolve_image)

VOCAB_PATH = SCRIPTS_DIR / "concept_vocab.json"
GRID_H, GRID_W = INPUT_H // 8, INPUT_W // 8          # 40 x 48


class ConceptSource:
    """Base class. Subclasses implement load() and score_image()."""
    name = "base"
    available = True
    unavailable_reason = ""

    def load(self, device):
        raise NotImplementedError

    def score_image(self, path, vocab):
        """Return (scores[C] np, raw_sims[C,2] np, maps[C,H0,W0] np).

        maps are at original image size; NaN marks cells where the model has
        no coverage (padding / cropped regions)."""
        raise NotImplementedError


class BioViLTSource(ConceptSource):
    """BioViL-T via health_multimodal (microsoft/BiomedVLP-BioViL-T).

    Fast path: image global + patch embeddings computed once per image, text
    embeddings once per prompt for the whole run (vs 20 engine calls/image).
    """
    name = "biovilt"

    def load(self, device):
        from health_multimodal.image.utils import ImageModelType, get_image_inference
        from health_multimodal.text.utils import BertEncoderType, get_bert_inference
        from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
        self.engine = ImageTextInferenceEngine(
            image_inference_engine=get_image_inference(ImageModelType.BIOVIL_T),
            text_inference_engine=get_bert_inference(BertEncoderType.BIOVIL_T_BERT))
        self.engine.to(device)
        self.device = device
        self._text_cache = {}          # prompt -> normalized [D] tensor
        return self

    def _text_emb(self, prompt: str):
        if prompt not in self._text_cache:
            e = self.engine.text_inference_engine.get_embeddings_from_prompt(
                prompt, normalize=False, verbose=False)
            self._text_cache[prompt] = F.normalize(e[0].float(), dim=0)
        return self._text_cache[prompt]

    def score_image(self, path, vocab, temp=0.1):
        path = Path(path)
        eng = self.engine
        img_emb = eng.image_inference_engine.get_projected_global_embedding(path)  # [D] L2-normed
        patches, (w, h) = eng.image_inference_engine.get_projected_patch_embeddings(path)  # [Hp,Wp,D]

        C = len(vocab)
        scores = np.zeros(C, dtype=np.float32)
        raw = np.zeros((C, 2), dtype=np.float32)
        maps = np.zeros((C, 0, 0), dtype=np.float32)
        first = True
        for i, c in enumerate(vocab):
            t_pos = self._text_emb(c["pos"])
            t_neg = self._text_emb(c["neg"])
            s_pos = float(img_emb @ t_pos)
            s_neg = float(img_emb @ t_neg)
            raw[i] = (s_pos, s_neg)
            scores[i] = torch.softmax(
                torch.tensor([s_pos, s_neg]) / temp, dim=0)[0].item()

            # patch-level similarity map for the positive prompt (same as
            # engine.get_similarity_map_from_raw_data, reusing cached embeddings)
            t_map = t_pos.reshape(1, -1)
            sim = eng._get_similarity_map_from_embeddings(patches, t_map)
            m = eng.convert_similarity_to_image_size(
                sim, width=w, height=h,
                resize_size=eng.image_inference_engine.resize_size,
                crop_size=eng.image_inference_engine.crop_size,
                val_img_transform=eng.image_inference_engine.transform,
                interpolation="bilinear")
            if first:
                maps = np.full((C, m.shape[0], m.shape[1]), np.nan, dtype=np.float32)
                first = False
            maps[i] = m
        return scores, raw, maps

    def reference_score(self, path, prompt):
        return self.engine.get_similarity_score_from_raw_data(Path(path), prompt)

    def reference_map(self, path, prompt):
        return self.engine.get_similarity_map_from_raw_data(
            Path(path), prompt, interpolation="bilinear")


class ReportConceptSource(ConceptSource):
    """Stub for future CheXpert Plus report-derived concept targets.

    Will parse radiology reports into per-image concept scores without needing
    BioViL-T at training time. Not implemented yet.
    """
    name = "report"
    available = False
    unavailable_reason = "CheXpert Plus reports not integrated yet"

    def load(self, device):
        raise NotImplementedError("ReportConceptSource is not implemented yet")

    def score_image(self, path, vocab):
        raise NotImplementedError("ReportConceptSource is not implemented yet")


CONCEPT_SOURCES = {
    "biovilt": BioViLTSource(),
    "report": ReportConceptSource(),
}


def get_source(spec: str) -> ConceptSource:
    if spec not in CONCEPT_SOURCES:
        raise KeyError(f"Unknown concept source {spec!r}. Known: {sorted(CONCEPT_SOURCES)}")
    src = CONCEPT_SOURCES[spec]
    if not src.available:
        raise RuntimeError(f"Concept source {spec!r} unavailable: {src.unavailable_reason}")
    return src


def cache_path_for(source: str) -> Path:
    return CACHE_DIR / f"concepts_{source}.pt"


def load_vocab(path=None):
    vocab = json.loads(Path(path or VOCAB_PATH).read_text(encoding="utf-8"))
    for c in vocab:
        for k in ("name", "pos", "neg", "label"):
            if k not in c:
                raise ValueError(f"vocab entry missing {k!r}: {c}")
    return vocab


def _maps_to_grid(maps_c_h0w0: np.ndarray):
    """[C,H0,W0] original-size maps -> (maps[C,40,48] fp16, map_valid[40,48]).

    Each map is aspect-fit into the 320x384 student frame (pad NaN), then
    area-downsampled to 40x48. map_valid is 1 where the cell is real image AND
    the majority of concepts produced a non-NaN value there."""
    C = maps_c_h0w0.shape[0]
    grid_maps = np.zeros((C, GRID_H, GRID_W), dtype=np.float32)
    coverage = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(C):
        canvas, _ = aspect_fit_float(maps_c_h0w0[i], INPUT_H, INPUT_W,
                                     pad_value=np.nan)
        filled = np.nan_to_num(canvas, nan=0.0)
        grid_maps[i] = cv2.resize(filled, (GRID_W, GRID_H),
                                  interpolation=cv2.INTER_AREA)
        has_val = (~np.isnan(canvas)).astype(np.float32)
        coverage += cv2.resize(has_val, (GRID_W, GRID_H),
                               interpolation=cv2.INTER_AREA)
    coverage /= C
    # coverage = fraction of concepts with a real (non-NaN) value in each cell.
    # aspect_fit_float pads with NaN, so padding and model-no-coverage both
    # count as invalid; coverage>0.5 => real image AND model produced a value.
    map_valid = (coverage > 0.5).astype(np.float32)
    return torch.from_numpy(grid_maps).half(), torch.from_numpy(map_valid)


def cmd_selftest(args):
    src = get_source(args.source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src.load(device)
    vocab = load_vocab(args.vocab)

    df, _, _ = load_labels(args.csv)
    paths = [resolve_image(canonical_key(p)) for p in df["Path"].head(3)]
    temp = args.temp
    worst_score, worst_map = 0.0, 0.0
    for path in paths:
        scores, raw, maps = src.score_image(path, vocab, temp=temp)
        for i, c in enumerate(vocab):
            ref_s_pos = src.reference_score(path, c["pos"])
            ref_s_neg = src.reference_score(path, c["neg"])
            ref_score = torch.softmax(
                torch.tensor([ref_s_pos, ref_s_neg]) / temp, dim=0)[0].item()
            d_score = abs(scores[i] - ref_score)
            ref_map = src.reference_map(path, c["pos"])
            d_map = float(np.nanmax(np.abs(
                np.nan_to_num(maps[i], nan=0.0) - np.nan_to_num(ref_map, nan=0.0))))
            worst_score = max(worst_score, d_score)
            worst_map = max(worst_map, d_map)
        print(f"  {path.name}: ok")
    print(f"Max |fast - reference| score={worst_score:.2e} map={worst_map:.2e}")
    if worst_score > 1e-4 or worst_map > 1e-4:
        raise AssertionError("Fast path does not match reference within 1e-4")
    print("SELFTEST PASSED")


def cmd_cache(args):
    from sklearn.metrics import roc_auc_score

    src = get_source(args.source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src.load(device)
    vocab = load_vocab(args.vocab)
    names = [c["name"] for c in vocab]

    df, targets, mask = load_labels(args.csv,
                                    blank_policy=args.blank_policy,
                                    uncertain_policy=args.uncertain_policy)
    keys = [canonical_key(p) for p in df["Path"]]
    out_path = args.output or cache_path_for(args.source)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = {"source": args.source, "vocab": vocab, "concepts": names,
            "temperature": args.temp, "grid_hw": (GRID_H, GRID_W),
            "scores": {}, "raw_sims": {}, "maps": {}, "map_valid": {}}
    if out_path.is_file():
        old = torch.load(out_path, map_location="cpu", weights_only=False)
        if old.get("source") == args.source and old.get("concepts") == names:
            for k_ in ("scores", "raw_sims", "maps", "map_valid"):
                blob[k_] = old.get(k_, {})
            print(f"Resuming: {len(blob['scores'])} keys already cached")

    todo = [k for k in keys if k not in blob["scores"]]
    print(f"{len(keys)} frontal images | {len(todo)} to process | {len(names)} concepts")

    for i, key in enumerate(todo):
        path = resolve_image(key)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot read {path}")
        scores, raw, maps = src.score_image(path, vocab, temp=args.temp)
        gmaps, gvalid = _maps_to_grid(maps)
        blob["scores"][key] = torch.from_numpy(scores)
        blob["raw_sims"][key] = torch.from_numpy(raw)
        blob["maps"][key] = gmaps
        blob["map_valid"][key] = gvalid
        if (i + 1) % 5 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
        if (i + 1) % 20 == 0:
            torch.save(blob, out_path)

    torch.save(blob, out_path)
    covered = sum(1 for k in keys if k in blob["scores"])
    print(f"Saved {out_path} | coverage {covered}/{len(keys)}")

    # sanity CSV: concept score vs related CheXpert label
    views = df["AP/PA"] if "AP/PA" in df.columns else pd_series_all(len(df))
    rows = []
    for j, c in enumerate(vocab):
        label = c["label"]
        if label not in TASKS:
            continue
        line = {"concept": c["name"], "label": label}
        for v in ["all", "AP", "PA"]:
            sel = np.arange(len(df)) if v == "all" else np.where(views == v)[0]
            y = targets[sel, TASKS.index(label)]
            s = np.array([blob["scores"][keys[k]][j].item() for k in sel])
            mm = mask[sel, TASKS.index(label)].astype(bool)
            y, s = y[mm], s[mm]
            auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
            line[f"auroc_{v}"] = auc
        rows.append(line)
        flag = "  <-- LOW" if (not np.isnan(line["auroc_all"]) and line["auroc_all"] < 0.6) else ""
        print(f"  {c['name']:32} ({label:16}) all={line['auroc_all']:.2f} "
              f"AP={line['auroc_AP']:.2f} PA={line['auroc_PA']:.2f}{flag}")
    import pandas as pd
    sanity_path = out_path.with_name(out_path.stem + "_sanity.csv")
    pd.DataFrame(rows).to_csv(sanity_path, index=False)
    print(f"Sanity table -> {sanity_path}")


def pd_series_all(n):
    import pandas as pd
    return pd.Series(["all"] * n)


def cmd_viz(args):
    src = get_source(args.source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src.load(device)
    vocab = load_vocab(args.vocab)

    key = canonical_key(args.key)
    path = resolve_image(key)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2 cannot decode {path}")
    scores, raw, maps = src.score_image(path, vocab, temp=args.temp)

    out_dir = CACHE_DIR / f"viz_{key.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # work in the student input frame so padding stays dark/aligned
    from common import INPUT_H as IH, INPUT_W as IW, aspect_fit
    img_fit = aspect_fit(img, IH, IW)
    base = cv2.cvtColor(img_fit, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(vocab):
        canvas, valid = aspect_fit_float(maps[i], IH, IW, pad_value=np.nan)
        filled = np.nan_to_num(canvas, nan=0.0)
        heat = cv2.normalize(filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = (heat * valid.astype(np.uint8)).astype(np.uint8)  # dark padding
        overlay = cv2.addWeighted(base, 0.6,
                                  cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.4, 0)
        out = out_dir / f"{i:02d}_{c['name']}.png"
        cv2.imwrite(str(out), overlay)
    print(f"Saved {len(vocab)} overlays -> {out_dir}")
    for i, c in enumerate(vocab):
        print(f"  {c['name']:32} score={scores[i]:.3f}")


def main():
    p = argparse.ArgumentParser(description="Concept source registry / cache CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("selftest", "cache", "viz"):
        c = sub.add_parser(name)
        c.add_argument("--source", default="biovilt")
        c.add_argument("--csv", type=Path,
                       default=SCRIPTS_DIR.parent / "Dataset" / "train_100.csv")
        c.add_argument("--vocab", type=Path, default=VOCAB_PATH)
        c.add_argument("--temp", type=float, default=0.1)
        c.add_argument("--output", type=Path, default=None)
        c.add_argument("--blank-policy", default="zero", choices=["zero", "mask"])
        c.add_argument("--uncertain-policy", default="mask",
                       choices=["mask", "ones", "zeros"])
        if name == "viz":
            c.add_argument("--key", required=True)
    args = p.parse_args()
    fn = {"list": lambda a: print("Concept sources:",
                                  ", ".join(f"{k}{'' if v.available else ' (unavailable)'}"
                                            for k, v in CONCEPT_SOURCES.items())),
          "selftest": cmd_selftest, "cache": cmd_cache, "viz": cmd_viz}[args.cmd]
    fn(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError,
            AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
