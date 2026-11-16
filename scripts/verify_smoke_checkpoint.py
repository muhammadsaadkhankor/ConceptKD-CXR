"""Reload supported student checkpoints and check every smoke-test image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from train_student import CheXpertDataset, LABELS, load_frontal, build_model, model_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-images", type=int, help="Test this many original validation frontal images instead of the 100-image smoke split")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert torch.cuda.is_available(), "CUDA required for this verification"
    report = {"device": "cuda", "gpu": torch.cuda.get_device_name(0), "checkpoints": []}
    for filename in ("student_best.pth", "student_last.pth"):
        checkpoint = torch.load(run / filename, map_location="cpu", weights_only=True)
        architectures = {"mobilenet_v3_small": "mobilenet_v3_small", "mobilenetv4_conv_small": "mobilenetv4_small",
                         "efficientnet_b0": "efficientnet_b0"}
        model_option = checkpoint.get("model_option", architectures[checkpoint["architecture"]])
        details = model_details(model_option, pretrained=False)
        assert checkpoint["architecture"] == details["architecture"]
        assert checkpoint.get("model_identifier", details["model_identifier"]) == details["model_identifier"]
        assert checkpoint["labels"] == LABELS
        model = build_model(model_option, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if filename == "student_last.pth":
            raw = torch.load(run / "student_weights.pth", map_location="cpu", weights_only=True)
            assert raw.keys() == checkpoint["state_dict"].keys()
            assert all(torch.equal(raw[k], checkpoint["state_dict"][k]) for k in raw)
            report["raw_weights_match_last_checkpoint"] = True
        model.cuda().eval()
        settings = checkpoint["preprocessing"]
        transform = transforms.Compose([
            transforms.Resize(tuple(settings["resize"])), transforms.ToTensor(),
            transforms.Normalize(settings["mean"], settings["std"]),
        ])
        if args.validation_images is not None:
            frame = load_frontal(Path(checkpoint["valid_csv"]), exact=not checkpoint.get("full_data", False))
            assert 0 < args.validation_images <= len(frame)
            frame = frame.iloc[:args.validation_images].copy()
        else:
            frames = [load_frontal(Path(checkpoint[key]), exact=True) for key in ("train_csv", "valid_csv")]
            frame = pd.concat(frames, ignore_index=True)
            assert len(frame) == 100 and frame["Path"].nunique() == 100
        expected_images = len(frame)
        dataset = CheXpertDataset(frame, Path(checkpoint["data_dir"]), transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        seen = 0
        minimum, maximum = float("inf"), -float("inf")
        with torch.inference_mode():
            for images, _, _ in loader:
                logits = model(images.cuda())
                assert logits.shape == (len(images), 5), tuple(logits.shape)
                assert torch.isfinite(logits).all().item(), "Nonfinite inference output"
                seen += len(images)
                minimum = min(minimum, logits.min().item())
                maximum = max(maximum, logits.max().item())
        torch.cuda.synchronize()
        assert seen == expected_images
        report["checkpoints"].append({
            "path": str(run / filename), "epoch": checkpoint["epoch"],
            "architecture": checkpoint["architecture"], "labels": checkpoint["labels"],
            "model_identifier": details["model_identifier"],
            "fresh_model": True, "strict_load": True, "images": seen,
            "logits_shape": [seen, 5], "all_finite": True,
            "minimum_logit": minimum, "maximum_logit": maximum,
        })
    (run / "reload_test.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
