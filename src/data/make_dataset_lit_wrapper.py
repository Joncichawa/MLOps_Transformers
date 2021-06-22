import pytorch_lightning as pl

from src.data.make_dataset import prepare_train_loaders


class DBPediaDataModule(pl.LightningDataModule):
    """DataLoader Class Overlay for Pytorch-Lightning"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.valid_loader, self.test_loader = prepare_train_loaders(
            self.config
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
