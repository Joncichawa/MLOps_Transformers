import torch

import src.data.make_dataset as mkd
from src.models.distil_bert_classifier import DistillBERTClass
from src.paths import MODELS_PATH


class Manager:
    def __init__(self, config, logger):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.loaded = False
        self.model = self._load_model(config)

    def _load_model(self, config):
        model_path = MODELS_PATH / self.config['model']['name']
        model = DistillBERTClass(config)
        if model_path.exists():
            self.logger.info('Loading saved model')
            model.load_state_dict(torch.load(str(model_path)))
        else:
            self.logger.info('Creating new model')
        model = model.to(self.device)
        return model

    def _save_model(self):
        model_path = MODELS_PATH / self.config['model']['name']
        torch.save(self.model.state_dict(), str(model_path))

    def _load_internal_dataset(self):
        if self.loaded:
            return None
        self.logger.info('Loading training dataset')
        self.train, self.valid = mkd.prepare_loaders(self.config)

    def _load_external_dataset(self, data):
        if self.loaded:
            return None
        return None  # TODO

    def _get_optimizer(self):
        option = self.config['model']['optimizer']
        if option == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.config['model']['lr'])
        else:
            raise Exception('Invalid optimizer name in experiment config file!')

    def train(self):
        self._load_internal_dataset()
        optimizer = self._get_optimizer()
        criterion = torch.nn.NLLLoss()
        epochs = self.config['model']['epochs']

        for e in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            for batch in self.train:
                ids = batch["input_ids"].unsqueeze(0).to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                self.model.train()  # Turn on dropout
                # Forward
                optimizer.zero_grad()
                output = self.model(ids, mask)
                output_ps = torch.exp(output)
                # Backward
                print(ids.shape)
                print(mask.shape)
                print(output.shape)
                print(labels.shape)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                running_loss += loss.item()
                running_acc += self.measure_acc(output_ps, labels)
            self.logger.info(f'========================[{e+1}/{epochs}]========================')
            self.logger.info(f'Training Loss: {running_loss / len(self.train)}')
            self.logger.info(f'Training Acc: {(running_acc / len(self.train)) * 100}%')
        self.logger.info('Training Finished!')

    def evaluate(self):
        self._load_internal_dataset()
        print('evaluate')

    def predict(self, data):
        data = self._load_external_dataset(data)
        print('predict')
