import pytorch_lightning as pl
import torch
import yaml

from src.models.lightning.model_manager import DBPediaDataModule
from src.models.lightning.model_wrapper_lit import LightningTextNet
from src.paths import EXPERIMENTS_PATH, MODELS_PATH


def load_config():
    CONFIG = EXPERIMENTS_PATH / 'hypersearch-large.yaml'
    with open(str(CONFIG)) as f:
        config = yaml.load(f)
    return config


def load_model(config):
    WEIGHTS = MODELS_PATH / 'hypersearch7.ckpt'
    model = LightningTextNet(
        config['model']['layers'],
        config['model']['dropout'],
        config['model']['lr'],
        config['model']['optimizer']
    ).load_from_checkpoint(checkpoint_path=str(WEIGHTS))
    model.eval()
    return model


def main():
    config = load_config()
    model = load_model(config)
    data = DBPediaDataModule(config)
    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None)
    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()
