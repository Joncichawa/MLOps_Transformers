import argparse
import logging
import warnings

import yaml

from src.models.model_manager import Manager
from src.paths import EXPERIMENTS_PATH

warnings.filterwarnings("ignore")


def setup_args():
    parser = argparse.ArgumentParser(
        description="Script for training, evaluation and inference of CNN MNIST model",
        usage="python main.py <command> <args>"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

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


def run_training(logger, config):
    logger.info('==========>>> Running training <<<==========')
    logger.info(f"Model: {config['model']['name']}")

    manager = Manager(config, logger)
    manager.train()

    logger.info('===============>>> Result <<<===============')


def run_evaluate(logger, config):
    logger.info('==========>>> Running evaluation <<<==========')
    logger.info(f"Model: {config['model']['name']}")

    manager = Manager(config, logger)
    manager.evaluate()

    logger.info('===============>>> Result <<<===============')


def run_predict(logger, config, data):
    logger.info('==========>>> Running inference <<<==========')
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f'Data: {data}')

    manager = Manager(config, logger)
    manager.predict(data)

    logger.info('===============>>> Result <<<===============')


if __name__ == "__main__":
    args = setup_args()
    logger = setup_logger()
    config = load_config(args.config)
    if args.command == 'predict':
        run_predict(logger, config, args.data)
    if args.command == 'evaluate':
        run_evaluate(logger, config)
    if args.command == 'train':
        run_training(logger, config)
