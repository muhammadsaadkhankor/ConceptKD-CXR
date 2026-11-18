"""Quick check that the BioViL-T concept model loads and scores one image.

Run inside the `prog` env:
    python test_concepts.py

Outputs: 2 concept scores printed + 2 heatmap PNGs saved next to this script.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from health_multimodal.image.utils import ImageModelType, get_image_inference
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

IMG = Path(r"D:\Master_Study\Course_Work\Second-year\Applied Machine Learning"
           r"\Project-detail\Dataset\train\patient41582\study5\view1_frontal.jpg")
OUT_DIR = Path(__file__).resolve().parent
TEMP = 0.1  # placeholder temperature for pos/neg softmax; tuned later

# name: (positive prompt, negative prompt)
CONCEPTS = {
    "blunted costophrenic angle": ("blunted costophrenic angle", "sharp costophrenic angles"),
    "enlarged cardiac silhouette": ("enlarged cardiac silhouette", "normal heart size"),
}


def main():
    if not IMG.is_file():
        raise FileNotFoundError(f"Image not found: {IMG}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading BioViL-T (first run downloads weights from Hugging Face)...")
    engine = ImageTextInferenceEngine(
        image_inference_engine=get_image_inference(ImageModelType.BIOVIL_T),
        text_inference_engine=get_bert_inference(BertEncoderType.BIOVIL_T_BERT),
    )
    engine.to(device)
    print("Loaded on", device)

    img = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)

    for name, (pos, neg) in CONCEPTS.items():
        s_pos = engine.get_similarity_score_from_raw_data(IMG, pos)
        s_neg = engine.get_similarity_score_from_raw_data(IMG, neg)
        score = torch.softmax(torch.tensor([s_pos, s_neg]) / TEMP, dim=0)[0].item()
        print(f"{name:30} pos={s_pos:.3f} neg={s_neg:.3f} score={score:.3f}")

        # Patch-level similarity heatmap for the positive prompt
        sim_map = engine.get_similarity_map_from_raw_data(IMG, pos, interpolation="bilinear")
        sim_map = np.nan_to_num(sim_map, nan=float(np.nanmin(sim_map))).astype(np.float32)
        heat = cv2.normalize(sim_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
        overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6,
                                  cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.4, 0)
        out_path = OUT_DIR / f"concept_{name.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), overlay)
        print("  heatmap ->", out_path)


if __name__ == "__main__":
    main()