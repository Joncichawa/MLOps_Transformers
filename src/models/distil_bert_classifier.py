import torch
from transformers import DistilBertModel

BERT_OUTPUT = 768
MODEL_OUTPUT = 14


class DistillBERTClass(torch.nn.Module):
    def __init__(self, config: dict):
        super(DistillBERTClass, self).__init__()
        self.pretrained_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        layers = []
        input_features = BERT_OUTPUT
        for dim in config['model']['layers']:
            layers.append(torch.nn.Linear(input_features, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(config['model']['dropout']))
            input_features = dim
        layers.append(torch.nn.Linear(input_features, MODEL_OUTPUT))
        self.lin_layers = torch.nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        output_1 = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        x = hidden_state[:, 0]
        x = self.lin_layers(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

    def freeze_pretrained_params(self):
        for param in self.pretrained_bert.parameters():
            param.requires_grad = False



# if __name__ == '__main__':
#     # USAGE EXAMPLE
#     config_path = EXPERIMENTS_PATH / "experiment-base.yaml"
#     with open(config_path) as f:
#         config = yaml.load(f)
#
#     train_loader, test_loader = prepare_loaders(config)
#     model = DistillBERTClass(config)
#     for x in test_loader:
#         ids = torch.tensor(x["input_ids"], dtype=torch.long).unsqueeze(0)
#         mask = torch.tensor(x["attention_mask"], dtype=torch.long)
#
#         out = model.forward(ids, mask)
