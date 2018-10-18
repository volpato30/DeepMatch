import logging.config
import tensorflow as tf
from inference_utils import get_optimized_weights
import config
from InputParser import TrainParser, InferenceParser
from solver import Solver, Inferencer
from inference_utils import compute_indentification_under_fdr
logger = logging.getLogger(__name__)


def train():
    solver = Solver()
    solver.solve('exp_result.log')


def prep_tfrecord():
    train_parser = TrainParser(spectrum_file=config.train_file)
    train_parser.convert_to_tfrecord(config.train_record_path)

    valid_parser = TrainParser(spectrum_file=config.valid_file)
    valid_parser.convert_to_tfrecord(config.valid_record_path)


def infer():
    fdr = 0.01
    inferencer = Inferencer()
    infer_parser = InferenceParser(spectrum_file=config.test_file)
    iterator = infer_parser.test_set_iterator()
    scan_list = inferencer.inference(test_iterator=iterator)

    key_logp = lambda x: x.logp_score
    target, num_scans = compute_indentification_under_fdr(scan_list, fdr, key_logp)
    logger.info(f"{target}/{num_scans} identified with PEAKS score")

    key_deep_match = lambda x: x.deep_match_score
    target, num_scans = compute_indentification_under_fdr(scan_list, fdr, key_deep_match)
    logger.info(f"{target}/{num_scans} identified with DeepMatch score")

    _ = get_optimized_weights(scan_list)




def main():
    if config.mode == 'train':
        train()
    elif config.mode == 'prep':
        prep_tfrecord()
    elif config.mode == 'infer':
        infer()
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
