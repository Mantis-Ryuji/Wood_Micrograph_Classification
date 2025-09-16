import json
import torch

from src.config_utils import load_config
from src.data_pipeline import build_dataloaders
from src.model import FaceWoodNet
from src.training import Trainer


def _infer_n_species(cfg, meta):
    if "N_SPE" in cfg and int(cfg["N_SPE"]) > 0:
        return int(cfg["N_SPE"])
    if isinstance(meta, dict) and "spe2id" in meta:
        return len(meta["spe2id"])
    if hasattr(meta, "spe2id"):
        return len(meta.spe2id)
    raise KeyError("n_species could not be inferred; please set model.n_species in config.yaml")


def main():
    cfg = load_config("config.yaml")

    # Data
    (train_loader, val_loader, test_loader), (ds_train, ds_val, ds_test), meta = build_dataloaders(
        h5_path=cfg["H5_PATH"],
        csv_path=cfg["CSV_PATH"],
        ratios=cfg["RATIOS"],
        batch_size=cfg["BATCH"],
        num_workers=cfg["WORKERS"],
        pin_memory=cfg["PIN_MEM"],
        seed=cfg["SEED"],
        input_size=cfg["INPUT_SIZE"],
        balanced_sampler=cfg["BALANCED_SAMPLER"]
    )

    n_species = _infer_n_species(cfg, meta)

    # Model
    model = FaceWoodNet(
        n_classes=n_species,
        backbone=cfg["BACKBONE"],
        backbone_weights=cfg["BACKBONE_WEIGHTS"],
        arc_s=cfg["ARC_S"],
        arc_m=cfg["ARC_M"],
        subcenters=cfg["ARC_SUBCENTERS"],
        use_center=cfg["USE_CENTER"],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        n_classes=n_species,
        base_lr=cfg["BASE_LR"],
        epochs=cfg["EPOCHS"],
        scheduler=cfg["SCHEDULER"],
        scheduler_type=cfg["SCHEDULER_TYPE"],
        scheduler_patience=cfg["SCHED_PATIENCE"],
        early_stop=cfg["EARLY_STOP"],
        monitor=cfg["MONITOR"],
        monitor_mode=cfg["MONITOR_MODE"],
        out_dir=cfg["OUT_DIR"],
        save_history=cfg["SAVE_HISTORY"],
        save_param=cfg["SAVE_PARAM"],
        hist_file=cfg["HIST_FILE"],
        ckpt_name=cfg["CKPT_NAME"],
        label_smoothing=cfg["LABEL_SMOOTH"],
        focal_gamma=cfg["FOCAL_GAMMA"],

        # CenterLoss（固定係数は保持するが、通常はWPを使用）
        lambda_center=cfg["LAMBDA_CENTER"],
        center_lr=cfg["CENTER_LR"],

        # Warm-ups
        margin_warmup_epochs=cfg["MARGIN_WARMUP_EPOCHS"],
        margin_m0=cfg["MARGIN_M0"],
        margin_m1=cfg["MARGIN_M1"],
        center_warmup_epochs=cfg["CENTER_WARMUP_EPOCHS"],
        center_lambda_final=cfg["CENTER_LAMBDA_FINAL"],

        # Scheduler extras
        T_0=cfg["T_0"],
        T_mult=cfg["T_MULT"],
        scheduler_threshold=cfg["SCHEDULER_THRESHOLD"],
        scheduler_cooldown=cfg["SCHEDULER_COOLDOWN"],
        scheduler_min_lr=cfg["SCHEDULER_MIN_LR"],

        # Optim 調整
        weight_decay=cfg["WEIGHT_DECAY"],
        backbone_lr_mult=cfg["BACKBONE_LR_MULT"],
        head_lr_mult=cfg["HEAD_LR_MULT"],
        backbone_freeze_epochs=cfg["BACKBONE_FREEZE_EPOCHS"],

        # EMA / TTA
        use_ema=cfg["USE_EMA"],
        ema_decay=cfg["EMA_DECAY"],
        tta=cfg["TTA"],
        device=None,
        verbose=cfg["VERBOSE"],
    )

    hist = trainer.fit(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)

    # testの結果を <out_dir>/test.json に保存
    path = f'{cfg["OUT_DIR"].rstrip("/")}/test.json'
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics}, f, indent=2, ensure_ascii=False)
    print(json.dumps({"test": test_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
