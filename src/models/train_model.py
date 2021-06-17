import torch
import yaml
from torch import nn, optim

from src.data.make_dataset import prepare_train_loaders
from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import EXPERIMENTS_PATH

device = 'cpu'  # TODO: GPU


# TODO: torch-metrics

def train(config: dict):
    train_loader, test_loader = prepare_train_loaders(config)
    model = DistillBERTClass(config)
    criterion = nn.NLLLoss()
    # freeze_LM_params() Do we need to? Maybe set it as a hyperparameter?
    optimizer = _choose_optimizer(config, model)
    # metrics = []

    for epoch in range(config["model"]["epochs"]):
        print(f"Epoch: {epoch + 1}")
        running_train_loss = 0
        for b in train_loader:
            optimizer.zero_grad()

            ids = b['input_ids'].to(device, dtype=torch.long)
            mask = b['attention_mask'].to(device, dtype=torch.long)
            labels = b['label'].to(device, dtype=torch.long)

            log_out = model(ids, mask)
            train_loss = criterion(log_out, labels)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
        print(running_train_loss)
        # metrics += compute_metrics(model, train_loss, test_loader, criterion)


def _freeze_DistilBERT_params():
    """
    Freezes DistilBERT weights
    """
    pass


def compute_metrics(model, train_loss, test_loader, criterion):
    train_loss
    # TODO: Finish, add some metrics
    model.eval()
    with torch.no_grad():
        running_loss = 0
        for b in test_loader:
            ids = b['input_ids'].to(device, dtype=torch.long)
            mask = b['attention_mask'].to(device, dtype=torch.long)
            labels = b['label'].to(device, dtype=torch.long)
            log_out = model(ids, mask)
            test_loss = criterion(log_out, labels)
            running_loss += test_loss.item()
    model.train()


def _choose_optimizer(_config: dict, model):
    if _config["model"]["optimizer"] == "Adam":
        return optim.Adam(model.parameters(), lr=_config["model"]["lr"])
    else:
        raise Exception("Unknown optimizer name")


if __name__ == '__main__':
    config_path = EXPERIMENTS_PATH / "experiment-base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train(config)
