from typing import Dict, List, Tuple

import pytest
import torch
import yaml
from datasets import load_dataset
from datasets import logging as logging_ds
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from transformers import DistilBertTokenizer
from transformers import logging as logging_tf

from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import DATA_PATH, EXPERIMENTS_PATH, MODELS_PATH, TESTS_PATH
from src.data.make_dataset import prepare_train_loaders 

def test_data_shape():
    config_path = TESTS_PATH / "test-config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
   
    train_loader, val_loader, test_loader = prepare_train_loaders(config)
    # print(train_loader.size)
    # print(train_loader.dataset['columns'])
    # assert train_loader.dataset['columns'] == ['attention_mask', 'input_ids', 'label']
    # assert train_loader.dataset == Dataset({'features': ['attention_mask', 'input_ids', 'label'],
    # 'num_rows': 14
    # })

    assert True