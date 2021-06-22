from src.models.lightning.model_manager import LightningTextNet
from src.paths import MODELS_PATH


def predict_model(config: dict, data):
    model = LightningTextNet.load_from_checkpoint(
        str(MODELS_PATH / f"{config['model']['name']}.ckpt"),
        config['model']['layers'],
        config['model']['dropout'],
        config['model']['lr'],
        config['model']['optimizer']
    )
    output = model(data)  # TODO but is it necessary before deployment?
    print(output)