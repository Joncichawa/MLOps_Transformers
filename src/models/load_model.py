import torch
import yaml
from torch import nn, optim

from src.data.make_dataset import prepare_train_loaders, predict_loader
from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import EXPERIMENTS_PATH, ROOT, MODELS_PATH
from src.models.lightning.model_wrapper_lit import LightningTextNet

if __name__ == '__main__':
    config_path = EXPERIMENTS_PATH / "experiment-final.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')    
    # model = LightningTextNet(config)
    model = LightningTextNet.load_from_checkpoint(
        str(MODELS_PATH/ f"{config['model']['name']}.ckpt")
    )
    # model.load_state_dict(torch.load(ROOT / "hypersearch.ckpt", map_location=device))
    model.eval()

    texts = [
        "If you want to work your glutes when doing the \
        split squat you need to adopt a wider (further forward) stance.",
        "Yeah just as long as your spine is neutral, you're good",
        "No wonder why my lower back was always in pain when doing this exercice."
    ]
    
    pred_loader = predict_loader(texts, config)
    dummy_input = next(iter(pred_loader))
    for x in pred_loader:
        ids = x['input_ids'].to(device, dtype=torch.long)
        mask = x['attention_mask'].to(device, dtype=torch.long)
    onnx_path =  "./model.onnx"
    torch.onnx.export(model, args=(ids, mask), f=onnx_path, input_names=["input_ids", "attention_mask"], output_names=["output_class"],verbose=False)