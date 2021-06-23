import pytorch_lightning as pl
import yaml

from src.models.lightning.train_model_lit import train_model
from src.paths import EXPERIMENTS_PATH

if __name__ == '__main__':
    # To activate wandb go to shell, run 'wandb login' and follow the instruction
    pl.seed_everything(seed=42)
    config_path = EXPERIMENTS_PATH / "hypersearch-large.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = train_model(config)
