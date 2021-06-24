from typing import Dict, List, Tuple

import pytest
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
import logging

from src.models.model_manager import Manager

from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import DATA_PATH, EXPERIMENTS_PATH, MODELS_PATH, TESTS_PATH
from src.data.make_dataset import prepare_train_loaders 


def test_model_layers():
    with open(str(TESTS_PATH / "test-config.yaml")) as f:
        config = yaml.safe_load(f)

    log_fmt = '[%(asctime)s]\t%(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    manager = Manager(config, logger)
    # Testamount of input nodes 
    assert manager.model.lin_layers[0].in_features == 768
    # Test proper dropout value
    assert manager.model.lin_layers[2].p == config['model']['dropout']

def test_model_trainig():
    with open(str(TESTS_PATH / "test-config.yaml")) as f:
        config = yaml.safe_load(f)

    log_fmt = '[%(asctime)s]\t%(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    manager = Manager(config, logger)
    # Test training loop
    try:
        manager.model.train()
    except Exception:
        assert False
    finally:
        assert True
