from __future__ import annotations
import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class ArcMarginProductSubcenter(nn.Module):
    """
    SubCenter-ArcFace
      - weight: [C*K, D], subcenters K for each class
      - forward:
          cos_all: [B, C, K] = (x @ W^T).view(B, C, K); cos = max_K cos_all
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, subcenters: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.K = int(max(1, subcenters))

        self.weight = nn.Parameter(torch.randn(out_features * self.K, in_features) * 0.01)
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def set_margin(self, m: float):
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,D], normalized
        W = l2_norm(self.weight, dim=1)
        cosine_all = F.linear(x, W)  # [B, C*K]
        if self.K > 1:
            B = cosine_all.size(0)
            cosine_all = cosine_all.view(B, self.out_features, self.K)
            cosine = cosine_all.max(dim=2).values  # [B,C]
        else:
            cosine = cosine_all

        if target is None:
            return self.s * cosine

        # margin
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        # optional: easy margin
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s
        return logits


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.center = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.center)

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # features: [B,D] (normalized or not)
        centers_batch = self.center.index_select(0, target)  # [B,D]
        return ((features - centers_batch).pow(2).sum(dim=1)).mean()


class FaceWoodNet(nn.Module):
    """
    ConvNeXt/ViT等のbackbone + BNネック + SubCenter-ArcFace (+ optional CenterLoss)
    """
    def __init__(
        self,
        n_classes: int,
        backbone: str = "convnextv2_tiny",
        backbone_weights: Optional[str] = None,  # timm pretrained tag or None
        backbone_img_size: Optional[int] = None,
        arc_s: float = 30.0,
        arc_m: float = 0.15,
        subcenters: int = 1,
        use_center: bool = True,
        feat_dim: Optional[int] = None,  # override if needed
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.backbone_name = backbone

        self.backbone = timm.create_model(backbone, pretrained=(backbone_weights is not None), img_size=backbone_img_size)
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(num_classes=0, global_pool="avg")
        

        self.feat_dim = feat_dim if feat_dim is not None else getattr(self.backbone, "num_features", 1024)
        self.bn = nn.BatchNorm1d(self.feat_dim, eps=1e-5, momentum=0.1, affine=True)
        self.head = ArcMarginProductSubcenter(self.feat_dim, self.n_classes, s=arc_s, m=arc_m, subcenters=subcenters)
        self.center_loss = CenterLoss(self.n_classes, self.feat_dim) if use_center else None

    @torch.no_grad()
    def _current_margin(self) -> float:
        return float(self.head.m)

    def set_margin(self, m: float):
        self.head.set_margin(m)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)              # [B, D]
        feats = self.bn(feats)
        feats = l2_norm(feats, dim=1)

        logits = self.head(feats, target)
        out = {"logits": logits, "emb": feats}

        if self.center_loss is not None and target is not None:
            out["center_loss"] = self.center_loss(feats, target)
        return out
