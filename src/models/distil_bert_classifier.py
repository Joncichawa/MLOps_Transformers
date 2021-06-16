import torch
from transformers import DistilBertModel

from src.train_config import HyperParameters


class DistillBERTClass(torch.nn.Module):
    def __init__(self, hyper_params: HyperParameters):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, hyper_params.HIDDEN_LAYER_SIZE)
        self.dropout = torch.nn.Dropout(hyper_params.DROPOUT_RATE)
        self.out = torch.nn.Linear(hyper_params.HIDDEN_LAYER_SIZE, 14)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        x = hidden_state[:, 0]
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        output = self.out(x)
        return output


# if __name__ == '__main__':
#     # USAGE EXAMPLE
#     hyper_params = HyperParameters()
#     train_loader, test_loader = prepare_loaders(hyper_params)
#     model = DistillBERTClass(hyper_params)
#     for x in test_loader:
#         ids = torch.tensor(x["input_ids"], dtype=torch.long).unsqueeze(0)
#         mask = torch.tensor(x["attention_mask"], dtype=torch.long)
#
#         out = model.forward(ids, mask)
