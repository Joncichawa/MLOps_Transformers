import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data.make_dataset_lit_wrapper import DBPediaDataModule
from src.models.lightning.model_wrapper_lit import LightningTextNet
from src.paths import MODELS_PATH


def train_model(config: dict):
    model = LightningTextNet(
        config['model']['layers'],
        config['model']['dropout'],
        config['model']['lr'],
        config['model']['optimizer']
    )
    datamodule = DBPediaDataModule(config)
    checkpoint = ModelCheckpoint(
        monitor='val/acc',
        dirpath=str(MODELS_PATH),
        filename=f"{config['model']['name']}",
        save_top_k=1
    )
    early_stopping = EarlyStopping(monitor='val/acc')
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        max_epochs=int(config['model']['epochs']),
        callbacks=[early_stopping, checkpoint]
    )
    trainer.fit(model, datamodule=datamodule)
