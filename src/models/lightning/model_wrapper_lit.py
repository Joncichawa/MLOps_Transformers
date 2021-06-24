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
        prec = pl.metrics.Precision(num_classes=14, average='macro')
        recall = pl.metrics.Recall(num_classes=14, average='macro')
        self.test_confusion = pl.metrics.ConfusionMatrix(num_classes=14)
        self.test_acc = acc.clone()
        self.val_acc = acc.clone()
        self.val_prec = prec.clone()
        self.val_recall = recall.clone()
        # self.train_loss = pl.metrics.Metric()

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
        self.log('train_loss_step', loss)
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
        self.log('val_loss_step', loss)
        self.log('val_acc_step', self.val_acc(pred, batch["label"]))
        self.log('val_recall_step', self.val_recall(pred, batch["label"]))
        self.log('val_prec_step', self.val_prec(pred, batch["label"]))

    def validation_epoch_end(self, batch):
        self.log("val_prec_epoch", self.val_prec.compute())
        self.log("val_recall_epoch", self.val_recall.compute())
        self.log("val_acc_epoch", self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        self.eval()
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        output = self(ids, mask)
        pred = torch.exp(output).argmax(dim=1)
        self.log('test_confusion', self.test_confusion(pred, batch["label"]))
        self.log('test_acc_step', self.test_acc(pred, batch["label"]))

    def test_epoch_end(self, batch):
        self.log("test_confusion", self.test_confusion.compute())
        self.log("test_acc_epoch", self.test_acc.compute())

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise Exception('Invalid optimizer name in experiment config file!')

        # scheduler = {'monitor': 'val_acc'}
        return [optimizer]
