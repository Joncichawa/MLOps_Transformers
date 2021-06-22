import optuna
import yaml
import torch
import shutil

from src.models.lightning.model_manager import LightningText_Net, DBPediaDataModule
import pytorch_lightning as pl
from src.paths import FIGURES_PATH, EXPERIMENTS_PATH

from optuna import visualization
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial, epochs, config):
    # TODO find better way to create custom trials
    epochs = epochs
    layers = trial.suggest_int("layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    hidden_dims = [
        trial.suggest_discrete_uniform(f"units_layer_{i}", 16, 512, 16) for i in range(layers)
    ]
    lr = trial.suggest_float("lr", 1e-4, 1)

    datamodule = DBPediaDataModule(config)
    model = LightningText_Net(hidden_dims, dropout, lr, optimizer='Adam')

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=epochs,
        # gradient_clip_val=0.1,
        gpus=-1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(
        layers=layers, dropout=dropout,
        hidden_dims=hidden_dims,
        lr=lr
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()


def start(name, epochs, time, n_trials):
    config = {
        'dataset': {
            'train_samples': 800,
            'val_samples': 100,
            'test_samples': 800
        },
        'model': {
            'name': name,
            'max_sentence_length': 512,
            'optimization': 'Adam',
            'epochs': int(epochs),
            'batch_size': 5,
            'layers': None,
            'dropout': None,
            'lr': None,
        }
    }
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    try:
        study.optimize(
            lambda trial: objective(trial, int(epochs), config),
            n_trials=int(n_trials),
            timeout=float(time)
        )
    except KeyboardInterrupt:
        output = input("Do you want to save model and figures? [Y/n] ")
        if output not in [None, 'Y', 'y']:
            return None
    save_dir = FIGURES_PATH / f'{name}'
    try:
        save_dir.mkdir(parents=False, exist_ok=True)
        visualization.plot_optimization_history(study).write_image(
            file=str(save_dir / 'optimization.png'), format='png')
        visualization.plot_param_importances(study).write_image(
            file=str(save_dir / 'hyperparameters.png'), format='png')
        config['model']['lr'] = study.best_params['lr']
        config['model']['dropout'] = study.best_params['dropout']
        config['model']['layers'] = [int(study.best_params[f'units_layer_{i}'])
                                     for i in range(study.best_params['layers'])]
        with open(str(EXPERIMENTS_PATH / f'{name}.yaml'), 'w+') as f:
            yaml.dump(config, f, indent=4)
    except Exception as e:
        save_dir = shutil.rmtree(str(save_dir))
        raise e
