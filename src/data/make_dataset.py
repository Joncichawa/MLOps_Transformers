from typing import Tuple

import torch
from datasets import load_dataset

from src.paths import DATA_PATH, MODELS_PATH
from src.train_config import HyperParameters
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel  # TODO: Use before DataLoader!


class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 14)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        x = hidden_state[:, 0]
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        output = self.out(x)
        return output


def prepare_loaders(_hyper_params: HyperParameters) -> Tuple[DataLoader, DataLoader]:
    # See https://huggingface.co/docs/datasets/loading_datasets.html#cache-directory
    # We don't need to save the dataset in data dir, as caching is handled by huggingface
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=MODELS_PATH)

    _test_loader = prepare_single_loader(_hyper_params, "test", tokenizer)

    return _test_loader, _test_loader


def prepare_single_loader(_hyper_params: HyperParameters, split: str, tokenizer):
    dataset = load_dataset("dbpedia_14", cache_dir=DATA_PATH / "raw", split=split)
    dataset = dataset.remove_columns(column_names="title")
    dataset = dataset.map(lambda e: tokenizer.encode_plus(e['content'], None,
                                                          add_special_tokens=True,
                                                          max_length=_hyper_params.MAX_SENTENCE_LENGTH,
                                                          pad_to_max_length=True,
                                                          return_token_type_ids=True,
                                                          truncation=True),
                          batched=True,
                          remove_columns=["content"])
    dataset.set_format(type='torch')
    dataset.save_to_disk(DATA_PATH / "processed" / f"{split}.pt")
    loader = DataLoader(dataset, batch_size=_hyper_params.BATCH_SIZE)
    return loader


if __name__ == '__main__':
    hyper_params = HyperParameters(BATCH_SIZE=32)
    train_loader, test_loader = prepare_loaders(hyper_params)
    model = DistillBERTClass()
    for x in test_loader:
        ids = torch.tensor(x["input_ids"], dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(x["attention_mask"], dtype=torch.long)

        out = model.forward(ids, mask)
