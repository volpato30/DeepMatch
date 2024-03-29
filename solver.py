import os
import tensorflow as tf
import numpy as np
import logging
import config
from InputParser import prepare_dataset_iterators, CandidatePeptideDM, Scan
from model import deep_match_scoring

logger = logging.getLogger(__name__)


class Solver(object):
    """Helper class for defining training loop and train"""

    def __init__(self):
        """
        building computation graph here
        """
        self.save_dir = config.save_dir
        self.num_epoch = config.num_epochs
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

        self.global_step = tf.train.get_or_create_global_step()

        if config.FLAGS.linear_cosine:
            # liner cosine decay
            self.learn_rate = tf.train.linear_cosine_decay(config.init_lr, self.global_step,
                                                           config.lr_decay_steps, beta=0.005)
        else:
            # piecewise constant
            self.learn_rate = tf.train.piecewise_constant(self.global_step, config.boundaries, config.values)

        if config.FLAGS.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learn_rate)
        elif config.FLAGS.optimizer == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(self.learn_rate)
        elif config.FLAGS.optimizer == 'momentum':
            self.opt = tf.train.MomentumOptimizer(self.learn_rate, 0.95)
        else:
            raise ValueError('optimizer should be adam sgd or momentum')

        next_element, self.training_init_op, self.validation_init_op = prepare_dataset_iterators(batch_size=64)

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

        logger.info(f"logits shape:")
        logger.info(f"{logits.get_shape()}")

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
        tf.summary.histogram('pos_logits', pos_logits)
        tf.summary.histogram('neg_logits', neg_logits)
        tf.summary.scalar("learn_rate", self.learn_rate)

        total_loss = self.ce_loss + self.weight_l2_norm

        # gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(total_loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            gradients,
            config.grad_clip)
        tf.summary.scalar("gradient_global_norm", norm)
        self.train_op = self.opt.apply_gradients(
            zip(clipped_gradients, params),
            global_step=self.global_step)

        # self.train_op = self.opt.minimize(self.ce_loss, global_step=self.global_step)
        # self.train_op = self.opt.minimize(total_loss, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()

        num_param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        logger.info(f"the model has {num_param} parameters")

    def solve(self, output_file: str):
        # scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
        best_valid_loss = float("inf")
        # TODO: make MoniterSession work
        # with tf.train.MonitoredTrainingSession(checkpoint_dir="./chkpoint",
        #                                        scaffold=scaffold,
        #                                        save_summaries_secs=30,
        #                                        save_checkpoint_secs=None,
        #                                        save_checkpoint_steps=None) as sess:
        saver = tf.train.Saver(max_to_keep=1)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            first_round_writer = tf.summary.FileWriter(os.path.join(self.save_dir, "train_summary", "first_pass"),
                                                       sess.graph)
            sess.run(init_op)
            sess.run(self.training_init_op)
            summary = sess.run(self.summary_op, feed_dict={self.keep_prob: config.keep_prob})
            first_round_writer.add_summary(summary, 1)
            first_round_writer.flush()
            logger.info("finished running the first pass through")
            first_round_writer.close()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'train_summary'), sess.graph)
            sess.run(init_op)
            for epoch in range(1, config.num_epochs + 1):
                sess.run(self.training_init_op)
                while True:
                    train_step = 0
                    try:
                        _, summary, gs = sess.run([self.train_op, self.summary_op, self.global_step],
                                                  feed_dict={self.keep_prob: config.keep_prob})
                        if train_step % config.summary_steps == 0:
                            train_writer.add_summary(summary, gs)
                            train_writer.flush()
                        train_step += 1
                    except tf.errors.OutOfRangeError:
                        break

                # after training one epoch, do validation
                sess.run(self.validation_init_op, feed_dict={self.keep_prob: 1.0})
                losses = []
                accs = []
                logger.debug(f"finish {epoch}th epoch, start validation")
                while True:
                    try:
                        loss, acc = sess.run([self.ce_loss, self.accuracy], feed_dict={self.keep_prob: 1.0})
                        losses.append(loss)
                        accs.append(acc)
                    except tf.errors.OutOfRangeError:
                        valid_loss = np.mean(losses)
                        valid_acc = np.mean(accs)
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            logger.info(f"{epoch}th epoch, achieve new best validation loss: {valid_loss}")
                            saver.save(sess, os.path.join(self.save_dir, "deepMatch.ckpt"),
                                       global_step=self.global_step)
                        logger.info(f"{epoch}th epoch, validation loss: {valid_loss}\tvalidation acc: {valid_acc}")
                        break

        with open(output_file, 'a') as f:
            param_string = f"{config.embed_dimension}\t" \
                           f"{config.lstm_output_dimension}\t" \
                           f"{config.spectral_hidden_dimension}\t" \
                           f"{config.FLAGS.init_lr}\t" \
                           f"{best_valid_loss}\n"
            f.write(param_string)


class Inferencer(object):
    def __init__(self):
        """
        building computation graph here
        """
        self.save_dir = config.save_dir
        self.sequence_placeholder = tf.placeholder(tf.int64, shape=(None, config.peptide_max_length),
                                                   name='sequence_placeholder')
        self.sequence_length_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='sequence_length_placeholder')
        self.ion_location_placeholder = tf.placeholder(tf.int64,
                                                       shape=(
                                                       None, config.peptide_max_length - 1, config.num_ion_combination),
                                                       name='ion_location_placeholder')
        self.spectrum_placeholder = tf.placeholder(tf.float32, shape=(None, config.M), name='spectrum_placeholder')

        self.logits = deep_match_scoring(self.sequence_placeholder,
                                         self.sequence_length_placeholder,
                                         self.ion_location_placeholder,
                                         self.spectrum_placeholder,
                                         1.0,
                                         reuse=False)

    def inference(self, test_iterator):
        final_scan_list = []
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(self.save_dir)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            for data in test_iterator:
                input_spectrum, sequence, sequence_length, ion_location_index, scans_list = data
                deep_match_scores = sess.run(self.logits, feed_dict={
                    self.sequence_placeholder: sequence,
                    self.sequence_length_placeholder: sequence_length,
                    self.ion_location_placeholder: ion_location_index,
                    self.spectrum_placeholder: input_spectrum,
                })
                ii = 0
                for scan in scans_list:
                    new_cp_list = []
                    for cp in scan.retrieved_peptides:
                        new_cp_list.append(CandidatePeptideDM(peptide_str=cp.peptide_str,
                                                              logp_score=cp.logp_score,
                                                              is_decoy=cp.is_decoy,
                                                              deep_match_score=deep_match_scores[ii])
                                           )
                        ii += 1
                    final_scan_list.append(Scan(scan_id=scan.scan_id, retrieved_peptides=new_cp_list))
                assert ii == len(deep_match_scores)
        return final_scan_list

