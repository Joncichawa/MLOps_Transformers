import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
from urllib.request import urlopen
import json
import time

import logging
import os
import sys
from datetime import datetime

from typing import Dict, List, Tuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import requests

# Run the model on the backend
d=os.path.dirname(os.path.abspath(__file__))
modelfile=os.path.join(d , 'model.onnx')
labelfile=os.path.join(d , 'labels.json')

session = onnxruntime.InferenceSession(modelfile)
np.set_printoptions(precision=None, suppress=True, formatter={'float_kind':'{:f}'.format})

# get the name of the inputs of the model
input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name  

def predict_loader(texts: List[str], config: dict) -> List[Dict[str, Tensor]]:
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", cache_dir="/cache"
    )

    token_text_list = list(
        map(
            lambda e: tokenizer.encode_plus(
                e,
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

def predict_class_from_text(input_text):
    config_path = "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    labels = ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder',
    'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
    'Plant', 'Album', 'Film', 'WrittenWork']

    pred_loader = predict_loader(input_text, config)
    start = time.time()

    results = []
    for x in pred_loader:
        ids = x["input_ids"]
        mask = x["attention_mask"]

        raw_result = session.run(None, {input_name1: ids.numpy(), input_name2: mask.numpy()})
        res = torch.exp(torch.tensor(raw_result))
        idx = np.argmax(res.squeeze().numpy()).astype(int)
        results.append((str(idx), labels[idx]))

    end = time.time()

    inference_time = np.round((end - start) * 1000, 2)

    response = {
            'created': datetime.utcnow().isoformat(),
            'predictions': results,
            'latency': inference_time,
    }
    
    logging.info(f'returning {response}')
    return response

if __name__ == '__main__':
    print(predict_class_from_text(sys.argv[1]))
