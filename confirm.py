import sys, os, cv2, torch, pandas as pd

BASE = r"D:\Master_Study\Course_Work\Second-year\Applied Machine Learning\Project-detail"
TEACHER_DIR = os.path.join(BASE, "Scripts", "teachers", "chest-xray-bench")  # adjust if different
CSV = os.path.join(BASE, "Scripts", "chexpert_subset.csv")                 # your 100-image subset
SCRIPTS_DIR = os.path.join(BASE, "Scripts")

sys.path.insert(0, TEACHER_DIR)
from modeling import load_model, preprocess

cwd = os.getcwd(); os.chdir(TEACHER_DIR)
model, cfg = load_model("convnext_tiny"); os.chdir(cwd)
model = model.cuda().eval()

def resolve(p):
    # image_path in the CSV is relative to the Scripts folder (e.g. ..\Dataset\train\...)
    return os.path.normpath(os.path.join(SCRIPTS_DIR, p))

df = pd.read_csv(CSV)
df = df[df["image_path"].str.contains("frontal")]    # frontal only

out, missing = {}, 0
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for p in df["image_path"]:
        img = cv2.imread(resolve(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1; continue
        out[p] = model(preprocess(img, cfg).cuda())[0].float().cpu()

torch.save({"tasks": cfg["tasks"], "logits": out},
           os.path.join(BASE, "Scripts", "teacher_logits_convnext_tiny.pt"))
print(f"Saved {len(out)} | missing {missing}")