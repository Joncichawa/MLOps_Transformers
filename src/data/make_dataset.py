from typing import Tuple

from datasets import load_dataset
from datasets import logging as logging_ds
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from transformers import logging as logging_tf

from src.paths import DATA_PATH, MODELS_PATH

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
    dataset.set_format(type="torch")

    dataset_path = DATA_PATH / "processed" / f"{split}"
    dataset.save_to_disk(dataset_path)

    loader = DataLoader(
        dataset, batch_size=config["model"]["batch_size"], shuffle=split == "train"
    )
    return loader


if __name__ == "__main__":
    # USAGE EXAMPLE
    train_loader, test_loader = prepare_loaders()
