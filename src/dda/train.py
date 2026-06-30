import logging

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from dda.config import resolve_output
from dda.data import XbdDamageDataModule
from dda.model import build_model

log = logging.getLogger(__name__)


def make_datamodule(cfg: DictConfig, data_pct: float) -> XbdDamageDataModule:
    return XbdDamageDataModule(
        repo_id=cfg.dataset_repo,
        dataset_splits=tuple(cfg.dataset_splits),
        img_size=cfg.img_size,
        use_pre=cfg.use_pre,
        augment_photometric=cfg.augment_photometric,
        val_events=tuple(cfg.val_events),
        test_events=tuple(cfg.test_events),
        val_frac=cfg.val_frac,
        data_pct=data_pct,
        batch_size=cfg.batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        seed=cfg.seed,
    )


def make_trainer(
    cfg: DictConfig, out_dir, max_epochs: int, monitor: str = "val/dmg_macro_f1"
) -> tuple[pl.Trainer, ModelCheckpoint]:
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "ckpts",
        filename="best-{epoch:02d}-{val/dmg_macro_f1:.4f}",
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor=monitor, mode="max", patience=cfg.early_stop_patience)
    progress = TQDMProgressBar(refresh_rate=0)
    csv_logger = CSVLogger(save_dir=str(out_dir), name="lightning")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=cfg.precision,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=cfg.grad_clip,
        callbacks=[ckpt_cb, early, progress],
        logger=csv_logger,
        log_every_n_steps=10,
        default_root_dir=str(out_dir),
        deterministic="warn",
    )
    return trainer, ckpt_cb


def train(cfg: DictConfig) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    out_dir = resolve_output(cfg)

    dm = make_datamodule(cfg, cfg.data_pct)
    dm.setup("fit")
    if cfg.class_weights is None:
        cfg.class_weights = dm.class_weights

    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    model = build_model(cfg)
    trainer, ckpt_cb = make_trainer(cfg, out_dir, cfg.max_epochs)

    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path="best")

    summary = {
        "best_ckpt": ckpt_cb.best_model_path,
        "best_val_dmg_f1": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score else None,
        "test": {k: float(v) for k, v in test_metrics[0].items()} if test_metrics else None,
        "output_dir": str(out_dir),
    }
    log.info("Run summary: %s", summary)
    return summary
