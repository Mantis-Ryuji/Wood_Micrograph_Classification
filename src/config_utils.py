# src/config_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml


def _require(obj: Dict[str, Any], path: List[str], missing: List[str]):
    cur = obj
    for i, k in enumerate(path):
        if not isinstance(cur, dict) or k not in cur:
            missing.append(".".join(path[: i + 1]))
            return None
        cur = cur[k]
    return cur


def _get(obj: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _tuple3(x: Any, ctx: str) -> Tuple[float, float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
        if abs(a + b + c - 1.0) > 1e-6:
            raise ValueError(f"[config] {ctx} must sum to 1.0, got {x}")
        return (a, b, c)
    raise ValueError(f"[config] {ctx} must be a list/tuple of length 3.")


def _to_bool(x: Any, ctx: str) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    raise ValueError(f"[config] {ctx} must be bool, got {type(x).__name__}")


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[config] not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    missing: List[str] = []

    # -------- data (required) --------
    data = _require(y, ["data"], missing) or {}
    h5_path   = _require(data, ["h5_path"], missing)
    csv_path  = _require(data, ["csv_path"], missing)
    split     = _require(data, ["split"], missing) or {}
    ratios    = _require(split, ["ratios"], missing)
    seed      = _require(split, ["seed"], missing)
    dl        = _require(data, ["dataloader"], missing) or {}
    batch_sz  = _require(dl, ["batch_size"], missing)
    n_workers = _require(dl, ["num_workers"], missing)
    pin_mem   = _require(dl, ["pin_memory"], missing)
    input_sz  = _require(data, ["input_size"], missing)

    # -------- model (required) --------
    model = _require(y, ["model"], missing) or {}
    n_spe = _require(model, ["n_species"], missing)
    backbone = _require(model, ["backbone"], missing)
    backbone_weights = _get(model, ["backbone_weights"], None)
    use_center = _require(model, ["use_center_loss"], missing)

    arc = _require(model, ["arcface"], missing) or {}
    s_val = _require(arc, ["s"], missing)
    m_val = _require(arc, ["m"], missing)
    subcenters = _get(arc, ["subcenters"], 1)

    # -------- loss (OPTIONAL: defaults) --------
    loss = _get(y, ["loss"], {}) or {}
    use_class_weights = _get(loss, ["use_class_weights"], False)
    label_smoothing   = _get(loss, ["label_smoothing"], 0.0)
    focal_gamma       = _get(loss, ["focal_gamma"], 2.0)
    lambda_center     = _get(loss, ["lambda_center"], 0.0)
    center_lr         = _get(loss, ["center_lr"], 0.01)

    # -------- training (required core) --------
    train = _require(y, ["training"], missing) or {}
    base_lr   = _require(train, ["base_lr"], missing)
    epochs    = _require(train, ["epochs"], missing)
    early_stop= _require(train, ["early_stop"], missing)
    scheduler = _require(train, ["scheduler"], missing)
    sched_ty  = _require(train, ["scheduler_type"], missing)
    sched_pat = _require(train, ["scheduler_patience"], missing)

    monitor   = _require(train, ["monitor"], missing)
    mon_mode  = _require(train, ["monitor_mode"], missing)
    verbose   = _require(train, ["verbose"], missing)
    out_dir   = _require(train, ["out_dir"], missing)
    hist_file = _require(train, ["history_filename"], missing)
    ckpt_name = _require(train, ["checkpoint_name"], missing)
    save_hist = _require(train, ["save_history"], missing)
    save_param= _require(train, ["save_param"], missing)

    # -------- scheduler extras (required by our code path) --------
    sched_thresh   = _require(train, ["scheduler_threshold"], missing)
    sched_cooldown = _require(train, ["scheduler_cooldown"], missing)
    sched_min_lr   = _require(train, ["scheduler_min_lr"], missing)
    T_0            = _require(train, ["T_0"], missing)
    T_mult         = _require(train, ["T_mult"], missing)

    # -------- warmups (required) --------
    m_warm   = _require(train, ["margin_warmup_epochs"], missing)
    m0       = _require(train, ["margin_m0"], missing)
    m1       = _require(train, ["margin_m1"], missing)
    c_warm   = _require(train, ["center_warmup_epochs"], missing)
    c_final  = _require(train, ["center_lambda_final"], missing)

    # -------- extra knobs (OPTIONAL) --------
    weight_decay       = _get(train, ["weight_decay"], 0.0)
    backbone_lr_mult   = _get(train, ["backbone_lr_mult"], 0.1)
    head_lr_mult       = _get(train, ["head_lr_mult"], 1.0)
    backbone_freeze_ep = _get(train, ["backbone_freeze_epochs"], 0)
    balanced_sampler   = _get(train, ["balanced_sampler"], False)
    use_ema            = _get(train, ["use_ema"], False)
    ema_decay          = _get(train, ["ema_decay"], 0.999)
    tta                = _get(train, ["tta"], False)

    if missing:
        # ここでは *本当に必須* なキーだけでエラーにする
        miss = []
        for k in missing:
            # loss.* など optional はスキップ（_getで埋める）
            if not k.startswith("loss"):
                miss.append(k)
        if miss:
            raise KeyError("[config] missing required keys:\n  - " + "\n  - ".join(miss))

    cfg: Dict[str, Any] = {}

    # data
    cfg["H5_PATH"]  = str(h5_path)
    cfg["CSV_PATH"] = str(csv_path)
    cfg["RATIOS"]   = _tuple3(ratios, "data.split.ratios")
    cfg["SEED"]     = int(seed)
    cfg["BATCH"]    = int(batch_sz)
    cfg["WORKERS"]  = int(n_workers)
    cfg["PIN_MEM"]  = _to_bool(pin_mem, "data.dataloader.pin_memory")
    cfg["INPUT_SIZE"] = int(input_sz)

    # model
    cfg["N_SPE"] = int(n_spe)
    cfg["BACKBONE"] = str(backbone)
    cfg["BACKBONE_WEIGHTS"] = None if backbone_weights is None else str(backbone_weights)
    cfg["USE_CENTER"] = _to_bool(use_center, "model.use_center_loss")
    cfg["ARC_S"] = float(s_val)
    cfg["ARC_M"] = float(m_val)
    cfg["ARC_SUBCENTERS"] = int(subcenters)

    # loss (optional with defaults)
    cfg["USE_CLASS_WEIGHTS"] = _to_bool(use_class_weights, "loss.use_class_weights") if isinstance(use_class_weights, (bool, int, float)) else False
    cfg["LABEL_SMOOTH"] = float(label_smoothing)
    cfg["FOCAL_GAMMA"]  = float(focal_gamma)
    cfg["LAMBDA_CENTER"] = float(lambda_center)
    cfg["CENTER_LR"]     = float(center_lr)

    # training core
    cfg["BASE_LR"] = float(base_lr)
    cfg["EPOCHS"]  = int(epochs)
    cfg["EARLY_STOP"] = _to_bool(early_stop, "training.early_stop")
    cfg["SCHEDULER"]  = _to_bool(scheduler, "training.scheduler")
    cfg["SCHEDULER_TYPE"] = str(sched_ty)
    cfg["SCHED_PATIENCE"] = int(sched_pat)

    cfg["MONITOR"] = str(monitor)
    cfg["MONITOR_MODE"] = str(mon_mode)
    cfg["VERBOSE"] = _to_bool(verbose, "training.verbose")
    cfg["OUT_DIR"] = str(out_dir)
    cfg["HIST_FILE"] = str(hist_file)
    cfg["CKPT_NAME"] = str(ckpt_name)
    cfg["SAVE_HISTORY"] = _to_bool(save_hist, "training.save_history")
    cfg["SAVE_PARAM"]   = _to_bool(save_param, "training.save_param")

    # scheduler extras
    cfg["SCHEDULER_THRESHOLD"] = float(sched_thresh)
    cfg["SCHEDULER_COOLDOWN"]  = int(sched_cooldown)
    cfg["SCHEDULER_MIN_LR"]    = float(sched_min_lr)
    cfg["T_0"]   = int(T_0)
    cfg["T_MULT"]= int(T_mult)

    # warmups
    cfg["MARGIN_WARMUP_EPOCHS"] = int(m_warm)
    cfg["MARGIN_M0"] = float(m0)
    cfg["MARGIN_M1"] = float(m1)
    cfg["CENTER_WARMUP_EPOCHS"] = int(c_warm)
    cfg["CENTER_LAMBDA_FINAL"]  = float(c_final)

    # extras
    cfg["WEIGHT_DECAY"] = float(weight_decay)
    cfg["BACKBONE_LR_MULT"] = float(backbone_lr_mult)
    cfg["HEAD_LR_MULT"] = float(head_lr_mult)
    cfg["BACKBONE_FREEZE_EPOCHS"] = int(backbone_freeze_ep)
    cfg["BALANCED_SAMPLER"] = _to_bool(balanced_sampler, "training.balanced_sampler")
    cfg["USE_EMA"] = _to_bool(use_ema, "training.use_ema")
    cfg["EMA_DECAY"] = float(ema_decay)
    cfg["TTA"] = _to_bool(tta, "training.tta")

    return cfg
