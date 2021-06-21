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


# def setup_module(module):
#     print('setup')
#     # make_dataset.main() check how to do this

# def teardown_module(module):
#     # todo
#     print('TEARDOWN')

# def test_dataloader():
#     config_path = "test-config.yaml"
#     with open(config_path) as f:
#         config = yaml.safe_load(f)



# def test_dummy_false2():
#     device = "cpu"
#     config_path = TESTS_PATH / "test-config.yaml"
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
   
#     train, test = prepare_train_loaders(config)
#     assert train.shape == 30522


def test_dummy_false():
    with pytest.raises(AssertionError):
        assert False
