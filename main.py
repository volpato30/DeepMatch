import logging.config
import tensorflow as tf
import config
from InputParser import TrainParser
from solver import Solver


def train():
    solver = Solver()
    solver.solve('exp_result.log')


def prep_tfrecord():
    train_parser = TrainParser(spectrum_file=config.train_file)
    train_parser.convert_to_tfrecord(config.train_record_path)

    valid_parser = TrainParser(spectrum_file=config.valid_file)
    valid_parser.convert_to_tfrecord(config.valid_record_path)


def main():
    if config.mode == 'train':
        train()
    elif config.mode == 'prep':
        prep_tfrecord()
    else:
        raise ValueError("not supported mode")


if __name__ == '__main__':
    log_file_name = 'deepMatch.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
