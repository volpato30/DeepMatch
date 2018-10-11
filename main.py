import logging.config
import tensorflow as tf
import config
from model import DeepMatchModel

def main():
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

    aa_sequence = tf.placeholder(tf.int64, shape=(None, config.peptide_max_length),
                                                  name='aa_sequence_placeholder')
    aa_sequence_length = tf.placeholder(tf.int64, shape=(None,),
                                                         name='aa_sequence_length_placeholder')
    ion_location_index = tf.placeholder(tf.int64,
                                                        shape=(None, config.peptide_max_length - 1,
                                                               config.num_ion_combination),
                                                        name="ion_location_index_placeholder")
    input_spectrum = tf.placeholder(tf.float32, shape=(None, config.M, 1),
                                                     name='input_spectrum_placeholder')

    model = DeepMatchModel(aa_sequence, aa_sequence_length, ion_location_index, input_spectrum)
    print(f"output logits shape: {model.output_logits.get_shape()}")


if __name__ == '__main__':
    main()