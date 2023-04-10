import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from dataloader import DataModule
from model import ColaModel
from visualization import VisualisationLogger

import hydra
from omegaconf import OmegaConf
import wandb
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")

    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="nlp-mlops-01")

    trainer = pl.Trainer(
        # default_root_dir="logs",
        devices=(1 if torch.cuda.is_available() else 0),
        max_epochs=cfg.training.max_epochs,
        # fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            VisualisationLogger(cola_data),
        ],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()
