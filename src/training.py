from __future__ import annotations

import os
import json
import math
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Metrics
# -------------------------
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> Tuple[float, ...]:
    """Return tuple of top-k accuracies."""
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()                                                # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))          # [maxk, B]
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / B).item())
        return tuple(res)


# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss (multi-class).
    Uses CE(reduction='none') * (1-p_t)^gamma and then reduce (mean/sum).
    Handles class weights and label smoothing through CE.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE per-sample
        ce = F.cross_entropy(
            logits, target,
            reduction="none",
            label_smoothing=self.label_smoothing
        )  # [B]

        # p_t
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1).clamp_(1e-8, 1.0)
        focal = (1.0 - pt).pow(self.gamma)

        loss = focal * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# -------------------------
# EMA
# -------------------------
class ModelEMA:
    """Exponential Moving Average of model parameters/buffers."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()  # same architecture
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        # initialize with current weights
        self.ema.load_state_dict(model.state_dict(), strict=True)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * d + msd[k] * (1.0 - d))

    def to(self, device: torch.device):
        self.ema.to(device)
        return self


# -------------------------
# EarlyStop helper
# -------------------------
@dataclass
class EarlyStop:
    best: Optional[float] = None
    best_epoch: int = -1
    counter: int = 0


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        base_lr: float,
        epochs: int,
        scheduler: bool,
        scheduler_type: str,
        scheduler_patience: int,
        early_stop: bool,
        monitor: str,
        monitor_mode: str,
        out_dir: str,
        save_history: bool,
        save_param: bool,
        hist_file: str,
        ckpt_name: str,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,

        # Center Loss
        lambda_center: float = 0.0,   # fixed add (kept for compatibility; usually use warmup below)
        center_lr: float = 0.01,

        # Warm-ups
        margin_warmup_epochs: int = 0,
        margin_m0: float = 0.20,
        margin_m1: float = 0.50,
        center_warmup_epochs: int = 0,
        center_lambda_final: float = 0.0,

        # Scheduler extras
        T_0: int = 10,
        T_mult: int = 2,
        scheduler_threshold: float = 1e-4,  # for ReduceLROnPlateau (kept for compat)
        scheduler_cooldown: int = 1,
        scheduler_min_lr: float = 1e-6,     # eta_min for Cosine

        # Optimization
        weight_decay: float = 0.0,
        backbone_lr_mult: float = 0.1,
        head_lr_mult: float = 1.0,
        backbone_freeze_epochs: int = 0,

        # EMA / TTA
        use_ema: bool = False,
        ema_decay: float = 0.999,
        tta: bool = False,

        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.n_classes = int(n_classes)
        self.base_lr = float(base_lr)
        self.epochs = int(epochs)
        self.scheduler = bool(scheduler)
        self.scheduler_type = str(scheduler_type)
        self.scheduler_patience = int(scheduler_patience)
        self.early_stop_flag = bool(early_stop)
        self.monitor = str(monitor)
        self.monitor_mode = str(monitor_mode)
        self.out_dir = str(out_dir)
        self.save_history = bool(save_history)
        self.save_param = bool(save_param)
        self.hist_file = str(hist_file)
        self.ckpt_name = str(ckpt_name)
        self.verbose = bool(verbose)

        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)

        self.lambda_center_fixed = float(lambda_center)
        self.center_lr = float(center_lr)

        self.margin_warmup_epochs = int(margin_warmup_epochs)
        self.margin_m0 = float(margin_m0)
        self.margin_m1 = float(margin_m1)
        self.center_warmup_epochs = int(center_warmup_epochs)
        self.center_lambda_final = float(center_lambda_final)

        self.T_0 = int(T_0)
        self.T_mult = int(T_mult)
        self.scheduler_threshold = float(scheduler_threshold)
        self.scheduler_cooldown = int(scheduler_cooldown)
        self.scheduler_min_lr = float(scheduler_min_lr)

        self.weight_decay = float(weight_decay)
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.head_lr_mult = float(head_lr_mult)
        self.backbone_freeze_epochs = int(backbone_freeze_epochs)

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.tta = bool(tta)

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.out_dir, exist_ok=True)

        self.model.to(self.device)

        # --------- Loss (Focal) ----------
        self.focal = FocalLoss(gamma=self.focal_gamma, label_smoothing=self.label_smoothing, reduction="mean")

        # --------- Optimizer param groups ----------
        # Center params (optional)
        center_params = []
        if getattr(self.model, "center", None) is not None:
            center_params = list(self.model.center.parameters())

        # Backbone params
        bb_params = list(self.model.backbone.parameters()) if getattr(self.model, "backbone", None) is not None else []

        # Head params (= the rest excluding center/backbone)
        center_ids = {id(p) for p in center_params}
        bb_ids = {id(p) for p in bb_params}
        head_params = [p for p in self.model.parameters() if p.requires_grad and (id(p) not in center_ids) and (id(p) not in bb_ids)]

        param_groups = []
        if bb_params:
            param_groups.append({"params": bb_params, "lr": self.base_lr * self.backbone_lr_mult, "weight_decay": self.weight_decay, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.base_lr * self.head_lr_mult, "weight_decay": self.weight_decay, "name": "head"})
        if center_params:
            # center has its own small lr and no weight decay
            param_groups.append({"params": center_params, "lr": self.center_lr, "weight_decay": 0.0, "name": "center"})

        self.optimizer = torch.optim.AdamW(param_groups)  # AdamW

        # --------- Scheduler ----------
        self.lr_scheduler = None
        if self.scheduler:
            if self.scheduler_type == "CosineWarmRestarts":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.scheduler_min_lr
                )
            elif self.scheduler_type == "ReduceLROnPlateau":
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode=self.monitor_mode, patience=self.scheduler_patience,
                    threshold=self.scheduler_threshold, cooldown=self.scheduler_cooldown,
                    factor=0.5, min_lr=self.scheduler_min_lr, verbose=self.verbose
                )
            else:
                raise ValueError(f"Unsupported scheduler_type: {self.scheduler_type}")

        # --------- EMA ----------
        self.ema = ModelEMA(self.model, decay=self.ema_decay).to(self.device) if self.use_ema else None

        # Early stop state
        self.es = EarlyStop()
        self.history = []

        # Backbone freeze (initial)
        self._apply_backbone_freeze(epoch=1)

        # Initialize ArcFace margin to m0
        if hasattr(self.model, "set_margin"):
            self.model.set_margin(self.margin_m0)

    # --------------------------
    def _current_margins(self, epoch: int) -> float:
        """Linear warmup for ArcFace margin m from m0 -> m1."""
        if self.margin_warmup_epochs <= 0:
            return self.margin_m1
        t = min(1.0, max(0.0, (epoch - 1) / float(self.margin_warmup_epochs)))
        return self.margin_m0 + t * (self.margin_m1 - self.margin_m0)

    def _current_center_lambda(self, epoch: int) -> float:
        """Linear warmup for center-loss coefficient 0 -> center_lambda_final."""
        if self.center_warmup_epochs <= 0:
            return self.center_lambda_final + self.lambda_center_fixed
        t = min(1.0, max(0.0, (epoch - 1) / float(self.center_warmup_epochs)))
        return self.lambda_center_fixed + t * self.center_lambda_final

    def _improved(self, curr: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if self.monitor_mode == "min":
            return curr < best
        elif self.monitor_mode == "max":
            return curr > best
        else:
            raise ValueError(f"Unknown monitor_mode: {self.monitor_mode}")

    def _apply_backbone_freeze(self, epoch: int):
        freeze = epoch <= max(0, self.backbone_freeze_epochs)
        if getattr(self.model, "backbone", None) is None:
            return
        for p in self.model.backbone.parameters():
            p.requires_grad_(not freeze)
        self._backbone_frozen = freeze

    def _lr_string(self) -> str:
        # pretty print each group lr by name
        groups = {g.get("name", f"g{idx}"): g["lr"] for idx, g in enumerate(self.optimizer.param_groups)}
        order = ["backbone", "head", "center"]
        show = []
        for k in order:
            if k in groups:
                show.append(f"{k}={groups[k]:.3e}")
        for k, v in groups.items():
            if k not in order:
                show.append(f"{k}={v:.3e}")
        return " | ".join(f"lr({s})" for s in show)

    # --------------------------
    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        # Early stop patience = scheduler_patience * 2 （提案どおり）
        early_patience = max(1, self.scheduler_patience * 2)

        for epoch in range(1, self.epochs + 1):
            # Warmups
            m_now = self._current_margins(epoch)
            if hasattr(self.model, "set_margin"):
                self.model.set_margin(m_now)

            lambda_center_now = self._current_center_lambda(epoch)

            # Freeze / unfreeze backbone
            self._apply_backbone_freeze(epoch)

            # Log header
            if self.verbose:
                bb_state = "FROZEN" if self._backbone_frozen else "TRAINABLE"
                print(f"\n[Epoch {epoch}/{self.epochs}] {self._lr_string()}  arcface m={m_now:.3f}  [backbone: {bb_state}]")

            # Train / Val
            train_m = self._run_epoch(train_loader, train=True, lambda_center=lambda_center_now)
            val_m   = self._run_epoch(val_loader,   train=False, lambda_center=lambda_center_now)

            # Scheduler step
            monitor_value = val_m.get(self.monitor)
            if monitor_value is None:
                raise KeyError(f"monitor key '{self.monitor}' not found; available={list(val_m.keys())}")

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(monitor_value)
                else:
                    # CosineWarmRestarts: step at epoch granularity is fine
                    self.lr_scheduler.step(epoch)

            # Early stop & checkpoint
            if self._improved(monitor_value, self.es.best):
                self.es.best = monitor_value
                self.es.best_epoch = epoch
                self.es.counter = 0
                if self.save_param:
                    self._save_ckpt(epoch)
                if self.verbose:
                    print(f"  ↳ New best {self.monitor}={monitor_value:.6f}")
            else:
                self.es.counter += 1
                if self.verbose:
                    print(f"  ↳ No improvement ({self.es.counter}/{early_patience}; early_stop at {early_patience})")

            # history
            rec = {"epoch": epoch}
            # also record per-group LR
            for i, g in enumerate(self.optimizer.param_groups):
                name = g.get("name", f"g{i}")
                rec[f"lr_{name}"] = float(g["lr"])
            rec.update(train_m); rec.update(val_m)
            self.history.append(rec)
            if self.save_history:
                self._dump_history()

            if self.early_stop_flag and self.es.counter >= early_patience:
                if self.verbose:
                    print(f"[EarlyStop] No improvement for {early_patience} epochs (>= {early_patience}). Stop at epoch {epoch}.")
                break

        return {"history": self.history, "best_epoch": self.es.best_epoch, "best_score": self.es.best}

    # --------------------------
    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, Any]:
        model_eval = self.ema.ema if self.ema is not None else self.model
        model_eval.eval().to(self.device)

        meters = {"loss": 0.0, "n": 0, "acc": 0.0, "top5": 0.0, "ce": 0.0}
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # TTA (eval only)
            logits = self._forward_tta(model_eval, x) if self.tta else model_eval(x)["logits"]

            # Use standard CE for reporting (no smoothing)
            ce = F.cross_entropy(logits, y, reduction="mean")
            B = x.size(0)
            acc1, acc5 = accuracy_topk(logits, y, topk=(1, 5))
            meters["loss"] += float(ce.item()) * B
            meters["ce"]   += float(ce.item()) * B
            meters["acc"]  += acc1 * B
            meters["top5"] += acc5 * B
            meters["n"]    += B

        n = max(1, meters["n"])
        return {"test_loss": meters["loss"] / n, "test_acc": meters["acc"] / n, "test_top5": meters["top5"] / n, "test_ce": meters["ce"] / n}

    # --------------------------
    def _forward_tta(self, model_eval: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # 4-way TTA: identity, hflip, rot90, rot270（軽量）
        outs = []
        outs.append(model_eval(x)["logits"])
        outs.append(model_eval(torch.flip(x, dims=[3]))["logits"])            # HFlip
        outs.append(model_eval(x.transpose(2, 3))["logits"])                   # rot90
        outs.append(model_eval(torch.flip(x.transpose(2, 3), dims=[3]))["logits"])  # rot270
        return torch.stack(outs, dim=0).mean(0)

    # --------------------------
    def _run_epoch(self, loader, train: bool, lambda_center: float) -> Dict[str, Any]:
        device = self.device
        self.model.train(mode=train)
        if not train:
            # eval uses BN in eval mode
            self.model.eval()

        meters = {"loss": 0.0, "n": 0, "acc": 0.0, "top5": 0.0}
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

                out = self.model(x, target=y)  # margin applied
                logits = out["logits"]
                loss = self.focal(logits, y)   # focal loss

                if "center_loss" in out and out["center_loss"] is not None and lambda_center > 0.0:
                    loss = loss + lambda_center * out["center_loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                if self.ema is not None:
                    self.ema.update(self.model)
            else:
                with torch.no_grad():
                    out = self.model(x, target=None)  # eval: no margin
                    logits = out["logits"]
                    loss = self.focal(logits, y)  # report focal for symmetry

            B = x.size(0)
            acc1, acc5 = accuracy_topk(logits, y, topk=(1, 5))
            meters["loss"] += float(loss.item()) * B
            meters["acc"]  += acc1 * B
            meters["top5"] += acc5 * B
            meters["n"]    += B

        n = max(1, meters["n"])
        prefix = "train_" if train else "val_"
        return {
            f"{prefix}loss": meters["loss"] / n,
            f"{prefix}acc": meters["acc"] / n,
            f"{prefix}top5": meters["top5"] / n,
        }

    # --------------------------
    def _save_ckpt(self, epoch: int):
        path = os.path.join(self.out_dir, self.ckpt_name)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_score": self.es.best,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
        }
        torch.save(payload, path)
        if self.verbose:
            print(f"[Trainer] Saved checkpoint to {path}")

    def _dump_history(self):
        path = os.path.join(self.out_dir, self.hist_file)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
