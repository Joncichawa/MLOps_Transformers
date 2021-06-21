import torch
import yaml
from torch import nn, optim

from src.data.make_dataset import prepare_train_loaders
from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import EXPERIMENTS_PATH

device = 'cpu'  # TODO: GPU


def train(config: dict):
    train_loader, val_loader, test_loader = prepare_train_loaders(config)
    model = DistillBERTClass(config)
    criterion = nn.NLLLoss()
    model.freeze_pretrained_params()
    optimizer = _choose_optimizer(config, model)
    metrics = []

    for epoch in range(config["model"]["epochs"]):
        print(f"Epoch: {epoch + 1}")
        running_train_loss = 0
        for t, b in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            ids = b['input_ids'].to(device, dtype=torch.long)
            mask = b['attention_mask'].to(device, dtype=torch.long)
            labels = b['label'].to(device, dtype=torch.long)

            log_out = model(ids, mask)
            train_loss = criterion(log_out, labels)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()
            print(f"Batch {t + 1} / {len(train_loader)} loss = ", train_loss.item())

        metrics += compute_metrics(model, running_train_loss, val_loader, criterion)
        print(running_train_loss)
    # TODO: evaluate on test loader


def compute_metrics(model, train_loss, test_loader, criterion) -> dict:
    model.eval()
    with torch.no_grad():
        all_equals = torch.empty((0, 1))
        running_test_loss = 0
        for t, b in enumerate(test_loader):
            ids = b['input_ids'].to(device, dtype=torch.long)
            mask = b['attention_mask'].to(device, dtype=torch.long)
            labels = b['label'].to(device, dtype=torch.long)
            log_ps = model(ids, mask)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(k=1)
            equals = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            all_equals = torch.cat([all_equals, equals])
            test_loss = criterion(log_ps, labels)

            running_test_loss += test_loss.item()

    test_accuracy = torch.mean(all_equals)
    metrics = {
        "train_loss": train_loss,
        "test_loss": running_test_loss,
        "accuracy_test": test_accuracy
    }
    print(metrics)
    return metrics


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
