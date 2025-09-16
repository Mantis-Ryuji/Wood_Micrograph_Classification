import os
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from src.config_utils import load_config
from src.data_pipeline import build_dataloaders
from src.model import FaceWoodNet


def topk_acc(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred)).any(dim=1)
    return correct.float().mean().item()


@torch.no_grad()
def forward_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = model(x.to(device, non_blocking=True), target=None)  # 推論パス
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out  # logits tensor を想定


@torch.no_grad()
def tta_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Flip + rot90 の 6-view TTA。各 view の logits を平均。
    """
    views = [
        x,                                           # id
        torch.flip(x, dims=[3]),                     # hflip
        torch.rot90(x, 1, dims=[2, 3]),              # rot90
        torch.rot90(x, 2, dims=[2, 3]),              # rot180
        torch.rot90(x, 3, dims=[2, 3]),              # rot270
        torch.flip(x, dims=[2]),                     # vflip
    ]
    logits_sum = None
    for v in views:
        lg = forward_logits(model, v, device)
        logits_sum = lg if logits_sum is None else (logits_sum + lg)
    return logits_sum / float(len(views))


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader,
             device: torch.device,
             num_classes: int) -> Dict[str, float]:
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()

    meters = {"loss": 0.0, "acc": 0.0, "top5": 0.0, "n": 0}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = tta_logits(model, x, device)  # ← TTA 強制ON
        loss = ce_loss(logits, y)

        B = x.size(0)
        meters["loss"] += float(loss.item()) * B
        meters["acc"]  += topk_acc(logits, y, k=1) * B
        meters["top5"] += topk_acc(logits, y, k=min(5, num_classes)) * B
        meters["n"]    += B

    n = max(1, meters["n"])
    return {
        "test_loss": meters["loss"] / n,
        "test_acc": meters["acc"] / n,
        "test_top5": meters["top5"] / n,
        "test_ce": meters["loss"] / n,
    }


def main():
    cfg = load_config("config.yaml")

    # --- Data ---
    (_, _, test_loader), (_, _, _), _ = build_dataloaders(
        h5_path=cfg["H5_PATH"],
        csv_path=cfg["CSV_PATH"],
        ratios=cfg["RATIOS"],
        batch_size=cfg["BATCH"],
        num_workers=cfg["WORKERS"],
        pin_memory=cfg["PIN_MEM"],
        seed=cfg["SEED"],
        input_size=cfg["INPUT_SIZE"],
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_species = int(cfg["N_SPE"])
    model = FaceWoodNet(
        n_classes=n_species,
        backbone=cfg["BACKBONE"],
        backbone_weights=cfg["BACKBONE_WEIGHTS"],
        arc_s=cfg["ARC_S"],
        arc_m=cfg["ARC_M"],
        subcenters=cfg["ARC_SUBCENTERS"],
        use_center=cfg["USE_CENTER"],  # 評価では center_loss は使わない
    ).to(device)

    # --- Load best checkpoint ---
    ckpt_path = os.path.join(cfg["OUT_DIR"], cfg["CKPT_NAME"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)

    # --- Evaluate with TTA (forced True) ---
    metrics = evaluate(model, test_loader, device, n_species)

    # --- Save JSON ---
    out_dir = Path(cfg["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_tta.json"
    payload = {
        "ckpt_path": ckpt_path,
        "ckpt_epoch": ckpt.get("epoch", None),
        "tta": True,  # 明示
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
