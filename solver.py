import tensorflow as tf
import config
from input_parser import prepare_dataset_iterators
from model import deep_match_scoring


class Solver(object):
    """Helper class for defining training loop and train"""
    def __init__(self):
        self.save_dir = config.save_dir
        self.num_epoch = config.num_epochs
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

        self.global_step = tf.train.get_or_create_global_step()
        self.learn_rate = tf.train.piecewise_constant(self.global_step, config.boundaries, config.values)
        self.opt = tf.train.AdamOptimizer(self.learn_rate)

        next_element, training_init_op, validation_init_op = prepare_dataset_iterators(batch_size=64)

        pos_logits = deep_match_scoring(next_element['pos_aa_sequence'],
                                        next_element['pos_aa_sequence_length'],
                                        next_element['pos_ion_location_index'],
                                        next_element['input_spectrum'],
                                        self.keep_prob,
                                        reuse=False)

        neg_logits = deep_match_scoring(next_element['neg_aa_sequence'],
                                        next_element['neg_aa_sequence_length'],
                                        next_element['neg_ion_location_index'],
                                        next_element['input_spectrum'],
                                        self.keep_prob,
                                        reuse=True)
        pos_target = tf.ones_like(pos_logits, dtype=tf.float32)
        neg_target = tf.zeros_like(neg_logits, dtype=tf.float32)
        logits = tf.concat((pos_logits, neg_logits), axis=0)
        target = tf.concat((pos_target, neg_target), axis=0)


        self.ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                               logits=logits,
                                                               name="compute_binary_crossentropy_loss")
        self.ce_loss = tf.reduce_mean(self.ce_loss)
        self.weight_l2_norm = tf.losses.get_regularization_loss()
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(target > 0.1,
                         logits > 0),
                tf.float32)
        )
        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('weight_l2_norm', self.weight_l2_norm)
        tf.summary.scalar('train_accuracy', self.accuracy)

    def solve(self):
        pass