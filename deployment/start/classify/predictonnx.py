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

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

# Run the model on the backend
d=os.path.dirname(os.path.abspath(__file__))
modelfile=os.path.join(d , 'model.onnx')
labelfile=os.path.join(d , 'labels.json')

session = onnxruntime.InferenceSession(modelfile)

# get the name of the first input of the model
input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name  
input_list = session.get_inputs() 
output_list = session.get_outputs()
output_name1 = session.get_outputs()[0].name

labels = load_labels(labelfile)

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
    
    logging.info(input_text)
    # arr1 = bytes(input_text, 'utf-8')
    # input_text = [" Gino Daniel Rea (born 18 September 1989 inTooting London) is an English motorcycle racer ofItalian descent.2010 was his first season in theSupersport World Championship riding for theIntermoto Czech Honda team. He won the 2009European Superstock 600 championship on a GeorgeWhite-backed Ten Kate Honda clinching the title bya single point in the final round at Portimão.Reawas a successful Motocross and Supermoto riderbefore switching to circuit racing in 2007."]
    # arr3 = bytes(input_text, 'utf-8')
    # logging.info(arr1)
    # logging.info(arr3)
    # logging.info(input_text)

    pred_loader = predict_loader(input_text, config)
    for x in pred_loader:
        ids = x["input_ids"]
        mask = x["attention_mask"]

    logging.info("########## RAW RESULT ZEROOO ########")
    raw_result = 0
    logging.info(raw_result)
    res = 0
    logging.info("########## RES AFTER EXP ########3")
    logging.info(res)

    start = time.time()
    raw_result = session.run(None, {input_name1: ids.numpy(), input_name2: mask.numpy()})
    end = time.time()
    np.set_printoptions(precision=None, suppress=True, formatter={'float_kind':'{:f}'.format})
    logging.info("########## RAW RESULT ########3")
    logging.info(raw_result)
    res = torch.exp(torch.tensor(raw_result))
    logging.info("########## RES AFTER EXP ########3")
    logging.info(res.numpy())

    inference_time = np.round((end - start) * 1000, 2)
    idx = np.argmax(res.squeeze().numpy())
    logging.info(res.squeeze().numpy())
    logging.info(idx)
    labels = ['Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder',
    'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
    'Plant', 'Album', 'Film', 'WrittenWork']

    response = {
            'created': datetime.utcnow().isoformat(),
            'prediction': labels[idx],
            'latency': inference_time,
    }
    logging.info(f'returning {response}')
    return response

if __name__ == '__main__':
    print(predict_class_from_text(sys.argv[1]))
