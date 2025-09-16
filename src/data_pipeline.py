from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# =========================================================
# ユーティリティ
# =========================================================
def build_label_map(df: pd.DataFrame) -> Dict[str, int]:
    # species のみを使用
    uniq = sorted(df["species"].unique())
    return {k: i for i, k in enumerate(uniq)}


def expand_split(split_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in split_df.iterrows():
        n = int(r["n_images"])
        for i in range(n):
            rows.append(
                dict(
                    dataset_path=r["dataset_path"],
                    idx_in_ds=i,
                    species=r["species"],
                    individual_id=r["individual_id"],
                )
            )
    return pd.DataFrame(rows)


# =========================================================
# picklable 変換（lambda 不使用）
# =========================================================
class RepeatTo3(nn.Module):
    """[1,H,W] -> [3,H,W] を repeat で実現"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(0) not in (1, 3):
            raise ValueError(f"RepeatTo3 expects [C,H,W] with C=1 or 3, got {tuple(x.shape)}")
        return x if x.size(0) == 3 else x.repeat(3, 1, 1)


class ResizeSquare(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W]
        C, H, W = x.shape
        if H == self.size and W == self.size:
            return x
        return F.interpolate(x.unsqueeze(0), size=(self.size, self.size),
                             mode="bilinear", align_corners=False).squeeze(0)


class RandomFlip90(nn.Module):
    """水平/垂直Flip と 90°回転（k=0..3）"""
    def __init__(self, p_h: float = 0.5, p_v: float = 0.5, p_rot: float = 0.75):
        super().__init__()
        self.p_h = float(p_h)
        self.p_v = float(p_v)
        self.p_rot = float(p_rot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_h:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < self.p_v:
            x = torch.flip(x, dims=[1])
        if torch.rand(1).item() < self.p_rot:
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
        return x


class NormalizeImagenet(nn.Module):
    """ConvNeXt/一般 ImageNet 正規化"""
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TrainTransform(nn.Module):
    """学習時前処理（軽量・ベクトル化前提）"""
    def __init__(self, size: int):
        super().__init__()
        ops: List[nn.Module] = [
            RepeatTo3(),
            RandomFlip90(0.5, 0.5, 0.75),
            ResizeSquare(size),
        ]
        ops.append(NormalizeImagenet())
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


class EvalTransform(nn.Module):
    """評価時前処理（TTAは Trainer 側）"""
    def __init__(self, size: int):
        super().__init__()
        ops: List[nn.Module] = [
            RepeatTo3(),
            ResizeSquare(size),
        ]
        ops.append(NormalizeImagenet())
        self.ops = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ops(x)


# =========================================================
# Dataset
# =========================================================
class WoodH5(Dataset):
    """
    HDF5:
      - 各 group の dataset は [N, H, W] (uint8, モノクロ)
    df_img:
      - expand_split で画像単位へ展開済み DataFrame
    返り値:
      x: Tensor [3, S, S] (float), y: species id (long)
    """
    def __init__(
        self,
        h5_path: str,
        df_img: pd.DataFrame,
        spe2id: Dict[str, int],
        input_size: int,
        train: bool
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.df = df_img.reset_index(drop=True)
        self.spe2id = spe2id
        self.input_size = int(input_size)
        self.train = bool(train)

        # 変換はトップレベル定義クラスのみ（Windows multiproc で picklable）
        self.transform = TrainTransform(self.input_size) if self.train \
                         else EvalTransform(self.input_size)

        # lazy open per-worker
        self._h5 = None

    def _ensure_h5(self):
        if self._h5 is None:
            # SWMR 読取専用
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        self._ensure_h5()
        r = self.df.iloc[i]
        ds = self._h5[r["dataset_path"]]            # [N, H, W] uint8
        x_np = ds[int(r["idx_in_ds"])]              # [H, W] uint8
        x = torch.from_numpy(x_np).float() / 255.0  # -> [H, W] float
        x = x.unsqueeze(0)                          # [1, H, W]
        x = self.transform(x)                       # [3, S, S]

        y = torch.tensor(self.spe2id[r["species"]], dtype=torch.long)
        return x, y

    def close(self):
        h = getattr(self, "_h5", None)
        if h is not None:
            try:
                h.close()
            except Exception:
                pass
            self._h5 = None

    def __del__(self):
        # 例外時でも安全に
        try:
            self.close()
        except Exception:
            pass


# =========================================================
# Sampler（クラス不均衡対策）
# =========================================================

def make_balanced_sampler(df_img: pd.DataFrame, spe2id: Dict[str, int], alpha: float = 0.5) -> WeightedRandomSampler:
    """
    alpha=1.0 → 完全バランス (1 / count)
    alpha=0.0 → 分布そのまま (weight=1)
    0 < alpha < 1 で緩やかに補正
    """
    counts = np.zeros(len(spe2id), dtype=np.int64)
    for sp in df_img["species"].values:
        counts[spe2id[sp]] += 1
    counts = np.maximum(counts, 1)

    # クラスごとの重み: 1 / (count ** alpha)
    w_per_class = 1.0 / (counts.astype(np.float64) ** alpha)
    w = np.array([w_per_class[spe2id[sp]] for sp in df_img["species"].values], dtype=np.float64)
    w_t = torch.as_tensor(w, dtype=torch.double)

    return WeightedRandomSampler(weights=w_t, num_samples=len(w), replacement=True)

# =========================================================
# スプリット
# =========================================================
from sklearn.model_selection import train_test_split


def split_by_individual(df: pd.DataFrame, ratios=(0.7, 0.2, 0.1), seed: int = 42):
    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    uniq = df["individual_id"].unique()
    train_ids, temp_ids = train_test_split(uniq, test_size=(1.0 - r_train), random_state=seed)
    frac_val = r_val / (r_val + r_test) if (r_val + r_test) > 0 else 0.5
    val_ids, test_ids = train_test_split(temp_ids, test_size=(1.0 - frac_val), random_state=seed)

    train_df = df[df["individual_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["individual_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["individual_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# =========================================================
# DataLoader エントリポイント
# =========================================================
def build_dataloaders(
    h5_path: str,
    csv_path: str,
    ratios=(0.7, 0.2, 0.1),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
    input_size: int = 320,
    balanced_sampler: bool = True,
):
    """
    返り値:
      (train_loader, val_loader, test_loader), (ds_train, ds_val, ds_test), meta
    """
    # CSV 読み込み
    df_full = pd.read_csv(csv_path)

    # 分割は individual_id ベース
    train_df, val_df, test_df = split_by_individual(df_full, ratios=ratios, seed=seed)

    # ラベル辞書（species）
    spe2id = build_label_map(df_full)

    # 画像行に展開
    df_train_img = expand_split(train_df)
    df_val_img   = expand_split(val_df)
    df_test_img  = expand_split(test_df)

    # Dataset
    ds_train = WoodH5(h5_path, df_train_img, spe2id, input_size, train=True)
    ds_val   = WoodH5(h5_path, df_val_img,   spe2id, input_size, train=False)
    ds_test  = WoodH5(h5_path, df_test_img,  spe2id, input_size, train=False)

    # Sampler / Loader
    sampler = make_balanced_sampler(df_train_img, spe2id) if balanced_sampler else None

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    meta = {
        "spe2id": spe2id,
        "n_species": len(spe2id),
        "n_train_imgs": len(ds_train),
        "n_val_imgs": len(ds_val),
        "n_test_imgs": len(ds_test),
    }
    return (train_loader, val_loader, test_loader), (ds_train, ds_val, ds_test), meta
