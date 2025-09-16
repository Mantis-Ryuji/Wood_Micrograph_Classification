import json
from pathlib import Path
import argparse
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
    row = df.loc[df["epoch"] == epoch]
    if row.empty:
        return None
    val = row[col].iloc[0]
    try:
        return float(val)
    except Exception:
        return None

def _fmt(v):
    return f"{v: .3f}" if (v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else "N/A"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ce", action="store_true", help="Accuracyパネルに CE(no margin) を右軸で重ねる")
    args = ap.parse_args()

    df = _load_history()
    test_meta, _ = _load_test_meta()

    # best(保存)エポック & test_tta
    best_epoch = None
    tta_loss = tta_acc = tta_top5 = None
    if isinstance(test_meta, dict):
        best_epoch = test_meta.get("ckpt_epoch", None)
        m = test_meta.get("metrics", {})
        tta_loss = m.get("test_loss", m.get("test_ce", None))
        tta_acc  = m.get("test_acc", None)
        tta_top5 = m.get("test_top5", None)

    last = {
        "train_acc":  _last_value(df, "train_acc"),
        "val_acc":    _last_value(df, "val_acc"),
        "train_top5": _last_value(df, "train_top5"),
        "val_top5":   _last_value(df, "val_top5"),
        "lr_backbone": _last_value(df, "lr_backbone"),
        "lr_head":     _last_value(df, "lr_head"),
        "val_ce":     _last_value(df, "val_ce") if "val_ce" in df.columns else None,
    }

    best_points = {
        "val_acc":  _get_at_epoch(df, "val_acc",  best_epoch) if best_epoch is not None else None,
        "val_top5": _get_at_epoch(df, "val_top5", best_epoch) if best_epoch is not None else None,
        "val_ce":   _get_at_epoch(df, "val_ce",   best_epoch) if (best_epoch is not None and "val_ce" in df.columns) else None,
    }

    # ---- Figure: 2x2 だが右下は削除（3枚表示） ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_acc, ax_top5, ax_lr, ax_empty = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    fig.delaxes(ax_empty)

    # (1) Top-1 Accuracy
    ax = ax_acc
    if "train_acc" in df.columns:
        ax.plot(df["epoch"], df["train_acc"], label=f"train ({_fmt(last['train_acc'])})")
    if "val_acc" in df.columns:
        ax.plot(df["epoch"], df["val_acc"],   label=f"val ({_fmt(last['val_acc'])})")
    if best_epoch is not None and best_points["val_acc"] is not None:
        ax.plot([best_epoch], [best_points["val_acc"]], "o", color="red", label=f"best@{best_epoch}", markersize=7)
    if tta_acc is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_acc], "o", color="blue", label=f"test_tta ({_fmt(tta_acc)})", markersize=7)
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Acc")
    ax.grid(True, alpha=0.3)
    legend1 = ax.legend(loc="lower right")

    # 右軸に CE をオーバーレイ（任意）
    if args.with_ce and ("val_ce" in df.columns):
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["val_ce"], linestyle="--", alpha=0.5, label=f"val_ce ({_fmt(last['val_ce'])})")
        if best_epoch is not None and best_points["val_ce"] is not None:
            ax2.plot([best_epoch], [best_points["val_ce"]], "o", color="gray", alpha=0.7, label="val_ce@best", markersize=6)
        ax2.set_ylabel("CE (no margin)")
        ax2.grid(False)
        # 2つ目の凡例
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc="upper right")

    # (2) Top-5 Accuracy
    ax = ax_top5
    any_curve = False
    if "train_top5" in df.columns:
        ax.plot(df["epoch"], df["train_top5"], label=f"train ({_fmt(last['train_top5'])})"); any_curve = True
    if "val_top5" in df.columns:
        ax.plot(df["epoch"], df["val_top5"],   label=f"val ({_fmt(last['val_top5'])})"); any_curve = True
    if best_epoch is not None and best_points["val_top5"] is not None:
        ax.plot([best_epoch], [best_points["val_top5"]], "o", color="red", label=f"best@{best_epoch}", markersize=7)
    if tta_top5 is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_top5], "o", color="blue", label=f"test_tta ({_fmt(tta_top5)})", markersize=7)
    ax.set_title("Top-5 Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Top-5 Acc")
    ax.grid(True, alpha=0.3)
    if any_curve: ax.legend(loc="lower right")

    # (3) Learning Rates
    ax = ax_lr
    if "lr_backbone" in df.columns:
        ax.plot(df["epoch"], df["lr_backbone"], label="lr_backbone")
    if "lr_head" in df.columns:
        ax.plot(df["epoch"], df["lr_head"],     label="lr_head")
    ax.set_title("Learning Rates")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = OUT_DIR / ("overview_training_tta_acc_lr.png" if not args.with_ce else "overview_training_tta_acc_lr_ce.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
