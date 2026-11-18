"""Teacher registry + CLI to cache teacher logits.

Examples (from Project-detail/Scripts):
    python teachers.py list
    python teachers.py cache --teacher xray-bench:convnext_tiny --csv ..\\Dataset\\train_100.csv
    python teachers.py cache --teacher xray-bench:densenet121_u_mixed --csv ..\\Dataset\\train_100.csv

Cache files land in Scripts/cache/teacher_<safe_name>.pt with logits keyed by
canonical_key (see common.py). Training never loads the teacher -- only the cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from common import (CACHE_DIR, DATA_DIR, SCRIPTS_DIR, TASKS, canonical_key,
                    load_labels, resolve_image)

TEACHERS_ROOT = SCRIPTS_DIR / "teachers" / "chest-xray-bench"
REPO_ID = "mamounyosef/chest-xray-bench"


class TeacherAdapter:
    """Base class. Subclasses implement load() and predict()."""
    name = "base"
    tasks = TASKS
    available = True
    unavailable_reason = ""

    def load(self, device):
        raise NotImplementedError

    def predict(self, img_gray_uint8) -> torch.Tensor:
        """Return logits tensor[len(tasks)] in self.tasks order."""
        raise NotImplementedError

    def reorder_to_tasks(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits from self.tasks order to common.TASKS order."""
        idx = [self.tasks.index(t) for t in TASKS]
        return logits[idx]


class ChestXrayBenchTeacher(TeacherAdapter):
    """Any folder from HF repo mamounyosef/chest-xray-bench (modeling.py API)."""

    def __init__(self, folder: str):
        self.folder = folder
        self.name = f"xray-bench:{folder}"
        self._model = None
        self._cfg = None
        self._mod = None

    def _ensure_downloaded(self):
        folder_dir = TEACHERS_ROOT / self.folder
        if (folder_dir / "config.json").is_file() and (TEACHERS_ROOT / "modeling.py").is_file():
            return
        from huggingface_hub import snapshot_download
        print(f"Downloading {self.folder} from {REPO_ID} ...")
        snapshot_download(REPO_ID,
                          allow_patterns=[f"{self.folder}/*", "modeling.py"],
                          local_dir=str(TEACHERS_ROOT))

    def load(self, device):
        self._ensure_downloaded()
        if str(TEACHERS_ROOT) not in sys.path:
            sys.path.insert(0, str(TEACHERS_ROOT))
        import modeling  # teachers/chest-xray-bench/modeling.py
        self._mod = modeling
        self._model, self._cfg = modeling.load_model(self.folder, device=str(device))
        self.tasks = list(self._cfg.get("tasks", TASKS))
        n_out = self._cfg.get("head", {}).get("n_logits")
        if n_out != len(TASKS):
            raise ValueError(
                f"{self.folder}: head has {n_out} logits, expected {len(TASKS)} "
                f"(one sigmoid logit per task). This folder is incompatible.")
        return self

    def predict(self, img_gray_uint8) -> torch.Tensor:
        import cv2
        if img_gray_uint8 is None:
            raise ValueError("image failed to load")
        x = self._mod.preprocess(img_gray_uint8, self._cfg).to(
            next(self._model.parameters()).device)
        with torch.no_grad():
            logits = self._model(x)
        return logits[0].float().cpu()


class XRVTeacher(TeacherAdapter):
    """torchxrayvision teacher. Only registered if torchxrayvision imports."""

    PATHOLOGY_MAP = {"Effusion": "Pleural Effusion"}

    def __init__(self, weights: str):
        self.weights = weights
        self.name = f"xrv:{weights}"

    def load(self, device):
        try:
            import torchxrayvision as xrv
        except ImportError as e:
            raise RuntimeError(
                "torchxrayvision is not installed; cannot use xrv teachers. "
                "pip install torchxrayvision") from e
        self._xrv = xrv
        self._device = device
        if self.weights == "jfhealthcare":
            from torchxrayvision.baseline_models import jfhealthcare
            self._model = jfhealthcare.DenseNet()
            self._model.eval().to(device)
            self.pathologies = list(jfhealthcare.DenseNet.pathologies) \
                if hasattr(jfhealthcare.DenseNet, "pathologies") else None
        else:
            self._model = xrv.models.DenseNet(weights=self.weights)
            self._model.eval().to(device)
            self.pathologies = list(xrv.models.DenseNet.pathologies)
        # map xrv pathology names -> TASKS
        self.tasks = [self.PATHOLOGY_MAP.get(p, p) for p in self.pathologies]
        missing = [t for t in TASKS if t not in self.tasks]
        if missing:
            raise ValueError(f"xrv:{self.weights} lacks tasks {missing}; "
                             f"has {self.tasks}")
        return self

    def predict(self, img_gray_uint8) -> torch.Tensor:
        import cv2
        import torch.nn.functional as F
        img = cv2.resize(img_gray_uint8, (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(255.0)
        x = x.repeat(3, 1, 1)[None].to(self._device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self._device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self._device)
        x = (x - mean) / std
        with torch.no_grad():
            out = self._model(x)[0].float().cpu()
        # some xrv models emit probabilities, others logits; logit(sigmoid(x))
        # recovers logits either way, so always round-trip through probabilities
        p = torch.sigmoid(out).clamp(1e-4, 1 - 1e-4)
        return torch.logit(p)


def _try_import_xrv():
    try:
        import torchxrayvision  # noqa: F401
        return True
    except Exception:
        return False


XRV_AVAILABLE = _try_import_xrv()

TEACHERS = {
    "xray-bench:convnext_tiny": ChestXrayBenchTeacher("convnext_tiny"),
    "xray-bench:rad_dino_vitB_768": ChestXrayBenchTeacher("rad_dino_vitB_768"),
    "xray-bench:medmae_vitb_nih_B_768_s2_seed1337":
        ChestXrayBenchTeacher("medmae_vitb_nih_B_768_s2_seed1337"),
    "xray-bench:convnext_base_22k_1600x1312":
        ChestXrayBenchTeacher("convnext_base_22k_1600x1312"),
    "xray-bench:densenet121_u_mixed": ChestXrayBenchTeacher("densenet121_u_mixed"),
    "xray-bench:densenet121_seed7": ChestXrayBenchTeacher("densenet121_seed7"),
}
if XRV_AVAILABLE:
    TEACHERS["xrv:chex"] = XRVTeacher("densenet121-res224-chex")
    TEACHERS["xrv:jfhealthcare"] = XRVTeacher("jfhealthcare")


def get_teacher(spec: str) -> TeacherAdapter:
    if spec not in TEACHERS:
        known = ", ".join(sorted(TEACHERS))
        raise KeyError(f"Unknown teacher {spec!r}. Known: {known}")
    return TEACHERS[spec]


def cache_path_for(spec: str) -> Path:
    safe = spec.replace(":", "_").replace("/", "_")
    return CACHE_DIR / f"teacher_{safe}.pt"


def cmd_list(_args):
    print("Available teachers:")
    for spec, t in sorted(TEACHERS.items()):
        print(f"  {spec}")
    if not XRV_AVAILABLE:
        print("  (xrv:* unavailable: torchxrayvision not installed)")


def cmd_cache(args):
    import cv2
    from sklearn.metrics import roc_auc_score

    teacher = get_teacher(args.teacher)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.load(device)
    print(f"Teacher {teacher.name} | tasks={teacher.tasks} | device={device}")

    df, targets, mask = load_labels(args.csv,
                                    blank_policy=args.blank_policy,
                                    uncertain_policy=args.uncertain_policy)
    out_path = args.output or cache_path_for(args.teacher)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blob = {"teacher": args.teacher, "tasks": TASKS, "logits": {}}
    if out_path.is_file():
        old = torch.load(out_path, map_location="cpu", weights_only=False)
        if old.get("teacher") == args.teacher:
            blob["logits"] = old.get("logits", {})
            print(f"Resuming: {len(blob['logits'])} keys already cached")

    keys = [canonical_key(p) for p in df["Path"]]
    todo = [(k, resolve_image(k)) for k in keys if k not in blob["logits"]]
    print(f"{len(keys)} frontal images | {len(todo)} to process")

    for i, (key, img_path) in enumerate(todo):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = teacher.predict(img)
            else:
                logits = teacher.predict(img)
        blob["logits"][key] = teacher.reorder_to_tasks(logits).float()
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  {i+1}/{len(todo)}", flush=True)
        # checkpoint every 20 images so a crash doesn't lose work
        if (i + 1) % 20 == 0:
            torch.save(blob, out_path)

    torch.save(blob, out_path)
    covered = sum(1 for k in keys if k in blob["logits"])
    print(f"Saved {out_path} | coverage {covered}/{len(keys)}")

    # sanity: teacher AUROC vs labels
    probs = np.stack([torch.sigmoid(blob["logits"][k]).numpy() for k in keys])
    m = mask.astype(bool)
    print("Teacher AUROC vs labels (blank=neg, uncertain masked):")
    for j, name in enumerate(TASKS):
        y, p, mm = targets[:, j], probs[:, j], m[:, j]
        y, p = y[mm], p[mm]
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        print(f"  {name:18} AUROC={auc:.3f}")


def main():
    p = argparse.ArgumentParser(description="Teacher registry / cache CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("cache")
    c.add_argument("--teacher", required=True)
    c.add_argument("--csv", type=Path, required=True)
    c.add_argument("--output", type=Path, default=None)
    c.add_argument("--blank-policy", default="zero", choices=["zero", "mask"])
    c.add_argument("--uncertain-policy", default="mask",
                   choices=["mask", "ones", "zeros"])
    args = p.parse_args()
    {"list": cmd_list, "cache": cmd_cache}[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
