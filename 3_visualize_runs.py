import json
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = [
    "epoch","train_loss","val_loss","train_ce","val_ce",
    "train_acc","val_acc","train_top5","val_top5",
    "lr_backbone","lr_head"
]

def _load_history():
    path = RUNS_DIR / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        hist = json.load(f)
    df = pd.DataFrame(hist)

    # 数値化 & ソート & 同一epoch重複は最後を採用
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)
    df = df.sort_values("epoch").groupby("epoch", as_index=False).last()
    return df

def _load_test_meta():
    for name in ("test_tta.json", "test.json"):
        p = RUNS_DIR / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta, name
    return None, None

def _get_at_epoch(df, col, epoch):
    if col not in df.columns or epoch is None:
        return None
    row = df.loc[df["epoch"] == epoch]
    if row.empty:
        return None
    val = row[col].iloc[-1]
    try:
        return float(val)
    except Exception:
        return None

def _fmt(v):
    return f"{v: .3f}" if (v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else "N/A"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-ce", action="store_true", help="Accuracyパネルに CE(no margin) を右軸で重ねる")
    ap.add_argument("--with-lr", action="store_true", help="LRパネルを表示する（デフォルト非表示）")
    args = ap.parse_args()

    df = _load_history()
    test_meta, _ = _load_test_meta()

    # best(保存)エポック決定
    best_epoch = None
    tta_loss = tta_acc = tta_top5 = None
    if isinstance(test_meta, dict):
        best_epoch = test_meta.get("ckpt_epoch", None)
        m = test_meta.get("metrics", {})
        tta_loss = m.get("test_loss", m.get("test_ce", None))
        tta_acc  = m.get("test_acc", None)
        tta_top5 = m.get("test_top5", None)

    # Fallback: test_meta が無ければ val_acc の最大エポック
    if best_epoch is None and "val_acc" in df.columns and not df["val_acc"].dropna().empty:
        best_epoch = int(df.loc[df["val_acc"].idxmax(), "epoch"])

    # 凡例に出す数値を best_epoch の値に差し替え（凡例の項目名は変更しない）
    show = {
        "train_acc":   _get_at_epoch(df, "train_acc",   best_epoch),
        "val_acc":     _get_at_epoch(df, "val_acc",     best_epoch),
        "train_top5":  _get_at_epoch(df, "train_top5",  best_epoch),
        "val_top5":    _get_at_epoch(df, "val_top5",    best_epoch),
        "lr_backbone": _get_at_epoch(df, "lr_backbone", best_epoch),
        "lr_head":     _get_at_epoch(df, "lr_head",     best_epoch),
        "val_ce":      _get_at_epoch(df, "val_ce",      best_epoch) if "val_ce" in df.columns else None,
    }

    # ---- Figure 構成：LRの有無でレイアウト切替 ----
    if args.with_lr:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax_acc, ax_top5, ax_lr, ax_empty = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
        fig.delaxes(ax_empty)
    else:
        fig, (ax_acc, ax_top5) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        ax_lr = None  # 使わない

    # (1) Top-1 Accuracy
    ax = ax_acc
    if "train_acc" in df.columns:
        ax.plot(df["epoch"], df["train_acc"], label=f"train ({_fmt(show['train_acc'])})")
    if "val_acc" in df.columns:
        ax.plot(df["epoch"], df["val_acc"],   label=f"val ({_fmt(show['val_acc'])})")
    if best_epoch is not None and show["val_acc"] is not None:
        ax.plot([best_epoch], [show["val_acc"]], "o", color="red", label=f"best@{best_epoch}", markersize=7)
    if tta_acc is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_acc], "o", color="blue", label=f"test_tta ({_fmt(tta_acc)})", markersize=7)
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Acc")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # 右軸に CE をオーバーレイ（任意）
    if args.with_ce and ("val_ce" in df.columns):
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["val_ce"], linestyle="--", alpha=0.5,
                 label=f"val_ce ({_fmt(show['val_ce'])})")
        if best_epoch is not None and show["val_ce"] is not None:
            ax2.plot([best_epoch], [show["val_ce"]], "o", color="gray", alpha=0.7, label="val_ce@best", markersize=6)
        ax2.set_ylabel("CE (no margin)")
        ax2.grid(False)
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc="upper right")

    # (2) Top-5 Accuracy
    ax = ax_top5
    any_curve = False
    if "train_top5" in df.columns:
        ax.plot(df["epoch"], df["train_top5"], label=f"train ({_fmt(show['train_top5'])})"); any_curve = True
    if "val_top5" in df.columns:
        ax.plot(df["epoch"], df["val_top5"],   label=f"val ({_fmt(show['val_top5'])})"); any_curve = True
    if best_epoch is not None and show["val_top5"] is not None:
        ax.plot([best_epoch], [show["val_top5"]], "o", color="red", label=f"best@{best_epoch}", markersize=7)
    if tta_top5 is not None:
        x_pos = best_epoch if best_epoch is not None else df["epoch"].iloc[-1]
        ax.plot([x_pos], [tta_top5], "o", color="blue", label=f"test_tta ({_fmt(tta_top5)})", markersize=7)
    ax.set_title("Top-5 Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Top-5 Acc")
    ax.grid(True, alpha=0.3)
    if any_curve: ax.legend(loc="lower right")

    # (3) Learning Rates（--with-lr のときだけ描画）
    if args.with_lr:
        ax = ax_lr
        if "lr_backbone" in df.columns:
            ax.plot(df["epoch"], df["lr_backbone"], label=f"lr_backbone")
        if "lr_head" in df.columns:
            ax.plot(df["epoch"], df["lr_head"],     label=f"lr_head")
        ax.set_title("Learning Rates")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 出力ファイル名
    fname = "training_tta_acc"
    if args.with_lr:
        fname += "_lr"
    if args.with_ce:
        fname += "_ce"
    out_path = OUT_DIR / f"{fname}.png"

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
