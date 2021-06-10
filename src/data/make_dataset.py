from typing import Tuple

from datasets import load_dataset
from src.train_config import HyperParameters
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer  # TODO: Use before DataLoader!


def prepare_loaders(_hyper_params: HyperParameters) -> Tuple[DataLoader, DataLoader]:
    # See https://huggingface.co/docs/datasets/loading_datasets.html#cache-directory
    # We don't need to save the dataset in data dir, as caching is handled by huggingface
    dataset_raw = load_dataset("dbpedia_14")
    dataset_raw = dataset_raw.remove_columns(column_names="title")

    train_dataset = dataset_raw["train"]
    test_dataset = dataset_raw["test"]

    train_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')

    _train_loader = DataLoader(train_dataset, batch_size=_hyper_params.BATCH_SIZE)
    _test_loader = DataLoader(test_dataset, batch_size=_hyper_params.BATCH_SIZE)

    return _train_loader, _test_loader


if __name__ == '__main__':
    hyper_params = HyperParameters(BATCH_SIZE=32)
    train_loader, test_loader = prepare_loaders(hyper_params)
    for x in test_loader:
        print()
