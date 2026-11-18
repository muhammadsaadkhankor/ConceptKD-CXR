"""Student registry: any timm CNN backbone + disease/concept heads.

Examples:
    from students import build_student, param_report
    model = build_student("efficientnet_b0.ra_in1k", num_concepts=10)

Student.forward(x) returns a dict with disease_logits, concept_logits,
concept_cam [B,C,h,w] and grid_hw -- enough for the spatial map loss and
Grad-CAM without a second forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from common import NUM_CLASSES

# name -> timm model name (all CNNs; ViTs rejected at build time)
STUDENTS = {
    "efficientnet_b0.ra_in1k": "efficientnet_b0.ra_in1k",
    "mobilenetv3_large_100": "mobilenetv3_large_100",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "convnext_nano": "convnext_nano",
}


def _is_vit(model) -> bool:
    cfg = getattr(model, "default_cfg", {}) or {}
    arch = str(cfg.get("architecture", "")).lower()
    return "vit" in arch or "vision_transformer" in arch or "swin" in arch


class Student(nn.Module):
    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES,
                 num_concepts: int = 10, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         num_classes=0)
        if _is_vit(self.encoder):
            raise ValueError(
                f"{backbone} is a ViT/transformer; it does not produce an NCHW "
                "feature grid needed for concept CAMs. Pick a CNN backbone.")

        # determine channel count + grid with a dummy forward (num_features
        # differs across backbones, e.g. MobileNetV3)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feat = self.encoder.forward_features(dummy)
        if feat.dim() != 4:
            raise ValueError(
                f"{backbone}: forward_features returned {feat.dim()}-D output, "
                "expected 4-D NCHW feature map (CNN backbone required).")
        feat_ch = feat.shape[1]          # grid_hw is input-dependent; see forward()

        self.disease_head = nn.Linear(feat_ch, num_classes)
        self.concept_head = nn.Linear(feat_ch, num_concepts)
        self.num_concepts = num_concepts

    def forward(self, x):
        feat = self.encoder.forward_features(x)        # [B,C,h,w]
        pooled = feat.mean(dim=(2, 3))                  # GAP == Linear head input
        disease_logits = self.disease_head(pooled)      # [B,5]
        concept_logits = self.concept_head(pooled)      # [B,C]
        # exact CAM: head is linear on GAP, so per-pixel score whose spatial
        # mean equals the logit
        W = self.concept_head.weight                   # [C, feat_ch]
        b = self.concept_head.bias                     # [C]
        concept_cam = torch.einsum("bchw,kc->bkhw", feat, W) + b[None, :, None, None]
        return {"disease_logits": disease_logits,
                "concept_logits": concept_logits,
                "concept_cam": concept_cam,
                "grid_hw": tuple(feat.shape[-2:])}


def build_student(backbone: str, num_classes: int = NUM_CLASSES,
                  num_concepts: int = 10, pretrained: bool = True) -> Student:
    if backbone not in STUDENTS:
        known = ", ".join(sorted(STUDENTS))
        raise KeyError(f"Unknown student {backbone!r}. Registered: {known}")
    return Student(STUDENTS[backbone], num_classes=num_classes,
                   num_concepts=num_concepts, pretrained=pretrained)


def param_report(model: Student):
    """(inference-path params, concept-head params) as ints."""
    n_inf = sum(p.numel() for p in model.encoder.parameters())
    n_inf += sum(p.numel() for p in model.disease_head.parameters())
    n_con = sum(p.numel() for p in model.concept_head.parameters())
    return n_inf, n_con
