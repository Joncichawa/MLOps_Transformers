from typing import Dict, List, Tuple

import torch
import yaml
from datasets import load_dataset
from datasets import logging as logging_ds
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding, DistilBertTokenizer
from transformers import logging as logging_tf

from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import DATA_PATH, EXPERIMENTS_PATH, MODELS_PATH

logging_tf.set_verbosity_error()  # to mute useless logging dump
logging_ds.set_verbosity_error()


def prepare_loaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    # See https://huggingface.co/docs/datasets/loading_datasets.html#cache-directory
    # We don't need to save the dataset in data dir, as caching is handled by huggingface
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir=MODELS_PATH
    )

    _test_loader = prepare_single_loader(config, "test", tokenizer)
    _train_loader = prepare_single_loader(config, "train", tokenizer)

    return _train_loader, _test_loader


def predict_loader(texts: List[str], config: dict) -> List[Dict[str, Tensor]]:
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir=MODELS_PATH
    )

    token_text_list = list(
        map(
            lambda e: tokenizer.encode_plus(
                e,
                None,
                add_special_tokens=True,
                max_length=config["model"]["max_sentence_length"],
                pad_to_max_length=True,
                truncation=True,
            ),
            texts,
        )
    )

    def f(e):
        return {
            "input_ids": torch.tensor(e.data["input_ids"], dtype=torch.long).unsqueeze(
                0
            ),
            "attention_mask": torch.tensor(
                e.data["attention_mask"], dtype=torch.long
            ).unsqueeze(0),
        }

    dataset = list(map(f, token_text_list))

    return dataset


def prepare_single_loader(config: dict, split: str, tokenizer):
    dataset = load_dataset("dbpedia_14", cache_dir=DATA_PATH / "raw", split=split)
    dataset = dataset.remove_columns(column_names="title")
    dataset = dataset.map(
        lambda e: tokenizer.encode_plus(
            e["content"],
            None,
            add_special_tokens=True,
            max_length=config["model"]["max_sentence_length"],
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        ),
        batched=True,
        remove_columns=["content"],
    )

    def f(e):
        return {
            "input_ids": torch.tensor(e["ids"], dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.tensor(e["mask"], dtype=torch.long).unsqueeze(0),
        }

    dataset = dataset.map(f, batched=True)

    dataset.set_format(type="torch")

    dataset_path = DATA_PATH / "processed" / f"{split}"
    dataset.save_to_disk(dataset_path)

    loader = DataLoader(
        dataset, batch_size=config["model"]["batch_size"], shuffle=split == "train"
    )
    return loader


if __name__ == "__main__":
    # USAGE EXAMPLE PREDICT
    config_path = EXPERIMENTS_PATH / "experiment-base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    texts = [
        "If you want to work your glutes when doing the split squat you need to adopt a wider (further forward) stance.",
        "Yeah just as long as your spine is neutral, you're good",
        "No wonder why my lower back was always in pain when doing this exercice.",
    ]

    test_loader = predict_loader(texts, config)
    model = DistillBERTClass(config)
    for x in test_loader:
        ids = x["input_ids"]
        mask = x["attention_mask"]

        out = model.forward(ids, mask)
        print(out)
