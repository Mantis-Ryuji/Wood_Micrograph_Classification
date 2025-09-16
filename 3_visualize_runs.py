import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_history():
    path = RUNS_DIR / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    df = pd.DataFrame(hist)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    return df

def _load_test_meta():
    for name in ("test_tta.json", "test.json"):
        p = RUNS_DIR / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta, name
    return None, None

def _last_value(df, col):
    return float(df[col].iloc[-1]) if col in df.columns and len(df[col]) > 0 else None

def _get_at_epoch(df, col, epoch):
    if col not in df.columns:
        return None
    # epoch は 1-based を想定
    row = df.loc[df["epoch"] == epoch]
    if row.empty:
        return None
    val = row[col].iloc[0]
    try:
        return float(val)
    except Exception:
        return None

def _fmt(v):
    return f"{v: .2f}" if (v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else "N/A"

def main():
    df = _load_history()
    test_meta, test_name = _load_test_meta()

    # ---- best (保存) エポック（赤丸） & test_tta（青丸） ----
    best_epoch = None
    tta_loss = tta_acc = tta_top5 = None
    if isinstance(test_meta, dict):
        best_epoch = test_meta.get("ckpt_epoch", None)
        m = test_meta.get("metrics", {})
        # loss は test_loss 優先、なければ test_ce
        tta_loss = m.get("test_loss", m.get("test_ce", None))
        tta_acc  = m.get("test_acc", None)
        tta_top5 = m.get("test_top5", None)

    # ---- 最終値（凡例用） ----
    last = {
        "train_loss": _last_value(df, "train_loss"),
        "val_loss":   _last_value(df, "val_loss"),
        "train_acc":  _last_value(df, "train_acc"),
        "val_acc":    _last_value(df, "val_acc"),
        "train_top5": _last_value(df, "train_top5"),
        "val_top5":   _last_value(df, "val_top5"),
        "lr_backbone": _last_value(df, "lr_backbone"),
        "lr_head":     _last_value(df, "lr_head"),
    }

    # ---- best_epoch 上の val 値（赤丸プロット位置） ----
    best_points = {
        "val_loss": _get_at_epoch(df, "val_loss", best_epoch) if best_epoch is not None else None,
        "val_acc":  _get_at_epoch(df, "val_acc",  best_epoch) if best_epoch is not None else None,
        "val_top5": _get_at_epoch(df, "val_top5", best_epoch) if best_epoch is not None else None,
    }

    # ---------------- Figure (2x2) ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # (1) Loss
    ax = axes[0, 0]
    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], label=f"train ({_fmt(last['train_loss'])})")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"],   label=f"val ({_fmt(last['val_loss'])})")
    # 赤丸: best（保存）エポックの val_loss
    if best_epoch is not None and best_points["val_loss"] is not None:
        ax.plot([best_epoch], [best_points["val_loss"]], "o", color="red",
                label=f"best@epoch {best_epoch}", markersize=8)
    # 青丸: test_tta の loss
    if tta_loss is not None:
        # 横位置は best_epoch に揃える（あるいは最終epochに置きたい場合は df['epoch'].iloc[-1] でもOK）
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_loss], "o", color="blue",
                label=f"test_tta ({_fmt(tta_loss)})", markersize=8)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (2) Accuracy
    ax = axes[0, 1]
    if "train_acc" in df.columns:
        ax.plot(df["epoch"], df["train_acc"], label=f"train ({_fmt(last['train_acc'])})")
    if "val_acc" in df.columns:
        ax.plot(df["epoch"], df["val_acc"],   label=f"val ({_fmt(last['val_acc'])})")
    if best_epoch is not None and best_points["val_acc"] is not None:
        ax.plot([best_epoch], [best_points["val_acc"]], "o", color="red",
                label=f"best@epoch {best_epoch}", markersize=8)
    if tta_acc is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_acc], "o", color="blue",
                label=f"test_tta ({_fmt(tta_acc)})", markersize=8)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Acc")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (3) Top-5 Accuracy
    ax = axes[1, 0]
    any_curve = False
    if "train_top5" in df.columns:
        ax.plot(df["epoch"], df["train_top5"], label=f"train ({_fmt(last['train_top5'])})"); any_curve = True
    if "val_top5" in df.columns:
        ax.plot(df["epoch"], df["val_top5"],   label=f"val ({_fmt(last['val_top5'])})"); any_curve = True
    if best_epoch is not None and best_points["val_top5"] is not None:
        ax.plot([best_epoch], [best_points["val_top5"]], "o", color="red",
                label=f"best@epoch {best_epoch}", markersize=8)
    if tta_top5 is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_top5], "o", color="blue",
                label=f"test_tta ({_fmt(tta_top5)})", markersize=8)
    ax.set_title("Top-5 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-5 Acc")
    ax.grid(True, alpha=0.3)
    if any_curve:
        ax.legend()

    # (4) Learning rates
    ax = axes[1, 1]
    if "lr_backbone" in df.columns:
        ax.plot(df["epoch"], df["lr_backbone"], label=f"lr_backbone")
    if "lr_head" in df.columns:
        ax.plot(df["epoch"], df["lr_head"],     label=f"lr_head")
    ax.set_title("Learning Rates")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = plt.gcf()
    out_path = OUT_DIR / "overview_training_tta.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
