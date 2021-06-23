import pytorch_lightning as pl
import torch

from src.models.distil_bert_classifier import DistillBERTClass


class LightningTextNet(pl.LightningModule):
    def __init__(self, layers, dropout, lr, optimizer, freeze_bert=True):
        super().__init__()
        self.model = DistillBERTClass.init_from_args(layers, dropout)
        if freeze_bert:
            self.model.freeze_pretrained_params()
        self.optimizer = optimizer
        self.lr = lr
        acc = pl.metrics.Accuracy()
        self.test_acc = acc.clone()
        self.val_acc = acc.clone()
        self.val_prec = pl.metrics.Precision(num_classes=14).clone()
        self.val_recall = pl.metrics.Recall(num_classes=14).clone()
        self.train_loss = pl.metrics.Metric()

        self.save_hyperparameters()

    def forward(self, ids, mask):
        return self.model(ids, mask)

    def training_step(self, batch, batch_idx):
        self.train()
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        criterion = torch.nn.NLLLoss()
        loss = criterion(output, batch["label"])
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        criterion = torch.nn.NLLLoss()
        loss = criterion(output, batch["label"])
        output_ps = torch.exp(output)
        pred = output_ps.argmax(dim=1)

        self.log('val/loss', loss)
        self.val_acc(pred, batch["label"])
        self.val_recall(pred, batch["label"])
        self.val_prec(pred, batch["label"])

    def validation_epoch_end(self, batch):
        prec = self.val_prec.compute()
        recall = self.val_recall.compute()
        accuracy = self.val_acc.compute()
        self.log("val/acc", accuracy)
        self.log("val/recall", recall)
        self.log("val/prec", prec)

    def test_step(self, batch, batch_idx):
        self.eval()
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        pred = output.argmax(dim=1, keepdim=True)
        self.test_acc.update(pred, batch["label"])

    def test_epoch_end(self, batch):
        self.log("test_acc", self.test_acc.compute())

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise Exception('Invalid optimizer name in experiment config file!')

        # scheduler = {'monitor': 'val_acc'}
        return [optimizer]
