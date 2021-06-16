from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertModel  # TODO: Use before DataLoader!
from transformers import DistilBertTokenizer

from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import DATA_PATH, MODELS_PATH
from src.train_config import HyperParameters


def prepare_loaders(_hyper_params: HyperParameters) -> Tuple[DataLoader, DataLoader]:
    # See https://huggingface.co/docs/datasets/loading_datasets.html#cache-directory
    # We don't need to save the dataset in data dir, as caching is handled by huggingface
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir=MODELS_PATH
    )

    _test_loader = prepare_single_loader(_hyper_params, "test", tokenizer)
    _train_loader = prepare_single_loader(_hyper_params, "train", tokenizer)

    return _train_loader, _test_loader


def prepare_single_loader(_hyper_params: HyperParameters, split: str, tokenizer):
    dataset = load_dataset("dbpedia_14", cache_dir=DATA_PATH / "raw", split=split)
    dataset = dataset.remove_columns(column_names="title")
    dataset = dataset.map(
        lambda e: tokenizer.encode_plus(
            e["content"],
            None,
            add_special_tokens=True,
            max_length=_hyper_params.MAX_SENTENCE_LENGTH,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        ),
        batched=True,
        remove_columns=["content"],
    )
    dataset.set_format(type="torch")

    dataset_path = DATA_PATH / "processed" / f"{split}"
    dataset.save_to_disk(dataset_path)

    loader = DataLoader(
        dataset, batch_size=_hyper_params.BATCH_SIZE, shuffle=split == "train"
    )
    return loader


if __name__ == "__main__":
    # USAGE EXAMPLE
    hyper_params = HyperParameters()
    train_loader, test_loader = prepare_loaders(hyper_params)
