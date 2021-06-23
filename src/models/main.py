import argparse
import logging
import warnings

import yaml

import src.models.lightning.experiments as experiments
from src.models.model_manager import Manager
from src.paths import EXPERIMENTS_PATH

warnings.filterwarnings("ignore")


def setup_args():
    parser = argparse.ArgumentParser(
        description="Script for training, evaluation and inference of CNN MNIST model",
        usage="python main.py <command> <args>"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    exp_parser = subparsers.add_parser('experiment')
    exp_parser.add_argument(
        '--name', required=True, help="Name of the best experiment to be saved")
    exp_parser.add_argument(
        '--trials', default=100, help="Max number of trials")
    exp_parser.add_argument(
        '--epochs', default=10, help="Max number of epochs per experiment")
    exp_parser.add_argument(
        '--time', default=60*60, help="Max time for experimentation to run in seconds")

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument(
        '--config', required=True, help="File name of an experiment in /experiments folder")

    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument(
        '--config', required=True, help="File name of an experiment in /experiments folder")

    pred_parser = subparsers.add_parser('predict')
    pred_parser.add_argument(
        '--config', required=True, help="File name of an experiment in /experiments folder")
    pred_parser.add_argument(
        '--data', required=True, help="Path to a file with .csv data")
    return parser.parse_args()


def setup_logger():
    log_fmt = '[%(asctime)s]\t%(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)


def load_config(name):
    with open(str(EXPERIMENTS_PATH / name)) as f:
        return yaml.load(f)


def run_experiments(logger, name, epochs, time, trials):
    logger.info('==========>>> Running experiments <<<==========')
    logger.info(f'Best Model: {name}')

    experiments.start(name, epochs, time, trials)

    logger.info('==========>>> Finished <<<==========')


def run_training(logger, config):
    logger.info('==========>>> Running training <<<==========')
    logger.info(f"Model: {config['model']['name']}")

    manager = Manager(config, logger)
    manager.train()

    logger.info('===============>>> Finished <<<===============')


def run_evaluate(logger, config):
    logger.info('==========>>> Running evaluation <<<==========')
    logger.info(f"Model: {config['model']['name']}")

    manager = Manager(config, logger)
    manager.evaluate()

    logger.info('===============>>> Finished <<<===============')


def run_predict(logger, config, data):
    logger.info('==========>>> Running inference <<<==========')
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f'Data: {data}')

    manager = Manager(config, logger)
    manager.predict(data)

    logger.info('===============>>> Finished <<<===============')


def main():
    args = setup_args()
    logger = setup_logger()
    if args.command == 'experiment':
        return run_experiments(logger, args.name, args.epochs, args.time, args.trials)

    config = load_config(args.config)
    if args.command == 'predict':
        return run_predict(logger, config, args.data)
    if args.command == 'evaluate':
        return run_evaluate(logger, config)
    if args.command == 'train':
        return run_training(logger, config)


if __name__ == "__main__":
    main()
