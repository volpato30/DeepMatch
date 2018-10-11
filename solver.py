import tensorflow as tf
import config
from model import DeepMatchModel


class Solver(object):
    def __init__(self, deep_match_model: DeepMatchModel):
        self.save_dir = config.save_dir
        self.num_epoch = config.num_epochs
        self.target_placeholder = tf.placeholder(tf.float32, shape=(None,), name='target_placeholder')

        self.global_step = tf.train.get_or_create_global_step()
        self.learn_rate = tf.train.piecewise_constant(self.global_step, config.boundaries, config.values)
        self.opt = tf.train.AdamOptimizer(self.learn_rate)
        self.model = deep_match_model

        self.ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_placeholder,
                                                               logits=self.model.output_logits,
                                                               name="compute_binary_crossentropy_loss")
        self.ce_loss = tf.reduce_mean(self.ce_loss)
        self.weight_l2_norm = tf.losses.get_regularization_loss()
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(self.target_placeholder > 0,
                         self.model.output_logits > 0),
                tf.float32)
        )
        tf.summary.scalar('ce_loss', self.ce_loss)
        tf.summary.scalar('weight_l2_norm', self.weight_l2_norm)
        tf.summary.scalar('train_accuracy', self.accuracy)

    def solve(self):
        pass