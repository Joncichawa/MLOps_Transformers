import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import src.data.make_dataset as mkd
from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import MODELS_PATH


class DBPediaDataModule(pl.LightningDataModule):
    """ DataLoader Class Overlay for Pytorch-Lightning """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_loader, self.valid_loader, self.test_loader = \
            mkd.prepare_train_loaders(self.config)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class LightningText_Net(pl.LightningModule):
    def __init__(self, layers, dropout, lr, optimizer, freeze_bert=True):
        super().__init__()
        self.model = DistillBERTClass.init_from_args(layers, dropout)
        if freeze_bert:
            self.model.freeze_pretrained_params()
        self.test_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, ids, mask):
        return self.model(ids, mask)

    def training_step(self, batch, batch_idx):
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        criterion = torch.nn.NLLLoss()
        loss = criterion(output, batch["label"])
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        # criterion = torch.nn.NLLLoss()
        # loss = criterion(output, batch["label"])
        # self.log('validation_loss', loss)
        output_ps = torch.exp(output)
        pred = output_ps.argmax(dim=1)
        self.val_acc.update(pred, batch["label"])

    def validation_epoch_end(self, batch):
        accuracy = self.val_acc.compute()
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx):
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        pred = output.argmax(dim=1, keepdim=True)
        self.test_acc.update(pred, batch["label"])

    def test_epoch_end(self, batch):
        self.log("test_acc", self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise Exception('Invalid optimizer name in experiment config file!')

        # scheduler = {'monitor': 'val_acc'}
        return [optimizer]


def train_model(config: dict):
    # Use this function for training the final model
    model = LightningText_Net(
        config['model']['layers'],
        config['model']['dropout'],
        config['model']['lr'],
        config['model']['optimizer']
    )
    datamodule = DBPediaDataModule(config)
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        dirpath=str(MODELS_PATH),
        filename=f"{config['model']['name']}",
        save_top_k=1
    )
    early_stopping = EarlyStopping(monitor='val_acc')
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        logger=True,
        max_epochs=int(config['model']['epochs']),
        callbacks=[early_stopping, checkpoint]
    )
    trainer.fit(model, datamodule=datamodule)


def predict_model(config: dict, data):
    model = LightningText_Net.load_from_checkpoint(
        str(MODELS_PATH / f"{config['model']['name']}.ckpt"),
        config['model']['layers'],
        config['model']['dropout'],
        config['model']['lr'],
        config['model']['optimizer']
    )
    output = model(data)  # TODO but is it necessary before deployment?
    print(output)
