import os, torch, cv2, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from health_multimodal.text.utils import get_bert_inference, BertEncoderType
from health_multimodal.image.utils import get_image_inference, ImageModelType
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

BASE = Path(r"D:\Master_Study\Course_Work\Second-year\Applied Machine Learning\Project-detail")
CSV = BASE / "Dataset" / "train_100.csv"
OUT = BASE / "Scripts" / "concept_targets_biovilt.pt"
TEMP = 0.1          # placeholder; tuned after the AUROC check
MAP_HW = (10, 12)   # stored heatmap size (H, W) for a later spatial loss

# name: (positive prompt, negative prompt, related CheXpert label)
CONCEPTS = {
    "enlarged_cardiac_silhouette": ("enlarged cardiac silhouette", "normal heart size", "Cardiomegaly"),
    "increased_cardiothoracic_ratio": ("increased cardiothoracic ratio", "normal cardiothoracic ratio", "Cardiomegaly"),
    "blunted_costophrenic_angle": ("blunted costophrenic angle", "sharp costophrenic angles", "Pleural Effusion"),
    "basal_fluid_meniscus": ("fluid meniscus at the lung base", "clear lung bases", "Pleural Effusion"),
    "air_bronchograms": ("air bronchograms", "no air bronchograms", "Consolidation"),
    "lobar_opacity": ("dense lobar opacity", "clear lungs without focal opacity", "Consolidation"),
    "perihilar_haziness": ("perihilar haziness", "clear perihilar regions", "Edema"),
    "interstitial_lines": ("interstitial thickening with Kerley B lines", "no interstitial thickening", "Edema"),
    "plate_like_opacity": ("linear plate-like opacity", "no linear opacity", "Atelectasis"),
    "volume_loss": ("lung volume loss with fissure shift", "normal lung volumes", "Atelectasis"),
}
NAMES = list(CONCEPTS)

def resolve(p):
    parts = str(p).replace("\\", "/").split("/")
    if parts[0] in {"CheXpert-v1.0-small", "CheXpert-v1.0"}:
        parts = parts[1:]
    return BASE / "Dataset" / Path(*parts)

engine = ImageTextInferenceEngine(get_image_inference(ImageModelType.BIOVIL_T),
                                  get_bert_inference(BertEncoderType.BIOVIL_T_BERT))
engine.to(torch.device("cuda"))

df = pd.read_csv(CSV)
df = df[df["Frontal/Lateral"].eq("Frontal")].reset_index(drop=True)

scores, maps, raw = {}, {}, {}
for i, p in enumerate(df["Path"]):
    img_path = resolve(p)
    s_row, m_row, r_row = [], [], []
    for name in NAMES:
        pos, neg, _ = CONCEPTS[name]
        s_pos = engine.get_similarity_score_from_raw_data(img_path, pos)
        s_neg = engine.get_similarity_score_from_raw_data(img_path, neg)
        s_row.append(torch.softmax(torch.tensor([s_pos, s_neg]) / TEMP, 0)[0].item())
        r_row.append([s_pos, s_neg])
        m = engine.get_similarity_map_from_raw_data(img_path, pos, interpolation="bilinear")
        m = np.nan_to_num(m, nan=float(np.nanmin(m)))
        m_row.append(cv2.resize(m.astype(np.float32), MAP_HW[::-1], interpolation=cv2.INTER_AREA))
    scores[p] = torch.tensor(s_row)
    raw[p] = torch.tensor(r_row)
    maps[p] = torch.from_numpy(np.stack(m_row)).half()
    print(f"{i+1}/{len(df)} done", end="\r")

torch.save({"concepts": NAMES, "prompts": CONCEPTS, "temperature": TEMP,
            "scores": scores, "raw_sims": raw, "maps": maps}, OUT)
print(f"\nSaved {len(scores)} images -> {OUT}")

# sanity check: concept score vs related CheXpert label
views = df["AP/PA"] if "AP/PA" in df.columns else pd.Series(["all"] * len(df))
for j, name in enumerate(NAMES):
    label = CONCEPTS[name][2]
    line = f"{name:32} ({label:16})"
    for v in ["all", "AP", "PA"]:
        sel = df.index if v == "all" else df.index[views.eq(v)]
        y = df.loc[sel, label].to_numpy(dtype=float)
        s = np.array([scores[df.loc[k, "Path"]][j].item() for k in sel])
        ok = np.isin(y, [0.0, 1.0])
        y, s = y[ok], s[ok]
        auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
        line += f" {v}={auc:.2f}"
    print(line)