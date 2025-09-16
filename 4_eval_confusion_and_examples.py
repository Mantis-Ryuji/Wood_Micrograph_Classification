from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import h5py
from sklearn.metrics import confusion_matrix, classification_report

# --- project imports ---
from src.config_utils import load_config
from src.data_pipeline import build_dataloaders
from src.model import FaceWoodNet

# =========================
# 設定
# =========================
USE_TTA = True
IMAGES_DIR = Path("results")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT = 5          # 少数クラス除外
N_EXAMPLES = 80         # 成功/失敗のサンプル数
PAGE_SIZE  = 16          # 4×4 固定

# =========================
# 軽量TTA
# =========================
@torch.no_grad()
def forward_tta(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    outs = [
        model(x)["logits"],
        model(torch.flip(x, dims=[3]))["logits"],
        model(x.transpose(2, 3))["logits"],
        model(torch.flip(x.transpose(2, 3), dims=[3]))["logits"]
    ]
    return torch.stack(outs, dim=0).mean(0)

def set_global_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# DataLoader から index も返すラッパー
class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, i

# =========================
# メイン
# =========================
def main():
    cfg = load_config("config.yaml")
    set_global_seeds(cfg["SEED"])

    # DataLoader 構築
    (_, _, _), (_, _, ds_test), meta = build_dataloaders(
        h5_path=cfg["H5_PATH"], csv_path=cfg["CSV_PATH"],
        ratios=cfg["RATIOS"], batch_size=cfg["BATCH"],
        num_workers=cfg["WORKERS"], pin_memory=cfg["PIN_MEM"],
        seed=cfg["SEED"], input_size=cfg["INPUT_SIZE"],
        balanced_sampler=cfg["BALANCED_SAMPLER"]
    )
    test_loader = DataLoader(
        IndexedDataset(ds_test),
        batch_size=cfg["BATCH"], shuffle=True,
        num_workers=cfg["WORKERS"], pin_memory=cfg["PIN_MEM"], drop_last=False
    )

    spe2id: Dict[str, int] = meta["spe2id"]
    id2spe: Dict[int, str] = {v: k for k, v in spe2id.items()}
    n_classes = meta["n_species"]

    # モデルロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceWoodNet(
        n_classes=n_classes,
        backbone=cfg["BACKBONE"], backbone_weights=cfg["BACKBONE_WEIGHTS"],
        arc_s=cfg["ARC_S"], arc_m=cfg["ARC_M"],
        subcenters=cfg["ARC_SUBCENTERS"], use_center=cfg["USE_CENTER"]
    ).to(device).eval()
    ckpt_path = Path(cfg["OUT_DIR"]) / cfg["CKPT_NAME"]
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    ckpt_epoch = int(payload.get("epoch", -1))

    # 推論
    y_true: List[int] = []; y_pred: List[int] = []; y_top5: List[bool] = []
    success_idx_set: set[int] = set(); failure_idx_set: set[int] = set()
    idx2tp: Dict[int, Tuple[int,int]] = {}

    raw_h5 = h5py.File(cfg["H5_PATH"], "r", libver="latest", swmr=True)

    with torch.no_grad():
        for xb, yb, ib in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = forward_tta(model, xb) if USE_TTA else model(xb)["logits"]
            pred = logits.argmax(dim=1)
            top5i = torch.topk(logits, k=min(5,n_classes), dim=1).indices
            top5h = (top5i == yb.view(-1,1)).any(dim=1)

            y_true.extend(yb.tolist()); y_pred.extend(pred.tolist()); y_top5.extend(top5h.tolist())
            for i in range(xb.size(0)):
                idx = int(ib[i])
                t, p = int(yb[i]), int(pred[i])
                idx2tp[idx] = (t, p)
                if t == p: success_idx_set.add(idx)
                else:      failure_idx_set.add(idx)

    y_true_np = np.array(y_true); y_pred_np = np.array(y_pred)
    top5_acc = float(np.mean(y_top5)) if y_top5 else float("nan")

    # 少数クラス除外
    counts = np.bincount(y_true_np, minlength=n_classes)
    kept_labels = [c for c in range(n_classes) if counts[c] >= MIN_SUPPORT]
    kept_names  = [id2spe.get(c, str(c)) for c in kept_labels]
    mask_keep = np.isin(y_true_np, kept_labels)
    y_true_k, y_pred_k = y_true_np[mask_keep], y_pred_np[mask_keep]

    # Confusion Matrix
    cm = confusion_matrix(y_true_k, y_pred_k, labels=kept_labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(10,len(kept_labels)*0.25), max(8,len(kept_labels)*0.25)))
    im = ax.imshow(cm_norm, interpolation="nearest", aspect="auto", cmap="Blues")
    ax.set_title(f"Confusion Matrix (test_dataset) [min_support={MIN_SUPPORT}]")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(kept_labels))); ax.set_yticks(np.arange(len(kept_labels)))
    ax.set_xticklabels(kept_names, rotation=90, fontsize=6); ax.set_yticklabels(kept_names, fontsize=6)
    fig.tight_layout(); fig.savefig(IMAGES_DIR/"confusion_matrix_norm_filtered.png", dpi=150); plt.close(fig)

    # Classification Report
    report = classification_report(y_true_k, y_pred_k, labels=kept_labels,
                                   target_names=kept_names, output_dict=True, zero_division=0)
    summary = {
        "ckpt_epoch": ckpt_epoch, "min_support": MIN_SUPPORT,
        "overall": {"acc_top1": float(np.mean(y_true_np==y_pred_np)), "acc_top5": top5_acc},
        "by_species": {n: report[n] for n in kept_names if n in report}
    }
    (IMAGES_DIR/"classification_report_filtered.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    # --- グリッド出力 ---
    rng = np.random.default_rng(cfg["SEED"])
    def _sample(idx_set:set[int], n:int)->List[int]:
        idxs=list(idx_set); 
        return idxs if len(idxs)<=n else list(rng.choice(idxs,n,replace=False))
    def _chunks(lst:List[int],size:int): return [lst[i:i+size] for i in range(0,len(lst),size)]
    def _draw(indices:List[int], title, prefix, fail):
        pages=_chunks(indices,PAGE_SIZE)
        for i,page in enumerate(pages):
            fig,axes=plt.subplots(4,4,figsize=(12,12)); axes=np.asarray(axes)
            fig.suptitle(f"{title} page {i+1}/{len(pages)}")
            for k in range(16):
                r,c=divmod(k,4); ax=axes[r,c]; ax.axis("off")
                if k>=len(page): continue
                idx=page[k]; rec=ds_test.df.iloc[idx]
                ds_path=rec["dataset_path"]; idx_in=int(rec["idx_in_ds"])
                img=raw_h5[ds_path][idx_in]; ax.imshow(img,cmap="gray",vmin=0,vmax=255)
                t,p=idx2tp[idx]; tn=id2spe.get(t,str(t)); pn=id2spe.get(p,str(p))
                ax.set_title(f"T:{tn}\nP:{pn}", fontsize=9, color=("red" if fail else "black"))
            plt.tight_layout(rect=[0,0.03,1,0.95])
            fig.savefig(IMAGES_DIR/f"{prefix}_{i}.png",dpi=150); plt.close(fig)

    succ=_sample(success_idx_set,N_EXAMPLES); fail=_sample(failure_idx_set,N_EXAMPLES)
    _draw(succ,"Success Examples","success_grid",fail=False)
    _draw(fail,"Failure Examples","failure_grid",fail=True)

    raw_h5.close()
    print("Done. See results/")

if __name__=="__main__":
    main()
