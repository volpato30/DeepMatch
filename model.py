import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
from tensorflow import layers
import config
from utils import batch_scatter
import logging

logger = logging.getLogger(__name__)


def bi_rnn(x, sequence_length, n_hidden, n_steps, reuse):
    """ helper function for building the bidirectional lstm
    :param x: [batch, n_steps, d] input tensor, float32
    :param sequence_length: [batch] int64
    :returns:
        outputs: a length T list of outputs
    """
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=reuse)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, reuse=reuse)
    # Get lstm cell output
    outputs, forward_final_state, backward_final_state = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                                      dtype=tf.float32,
                                                                                      sequence_length=sequence_length)

    return outputs, forward_final_state, backward_final_state


kernel_regularizer = tf.contrib.layers.l2_regularizer(config.weight_decay)


def vgg_1d(input_tensor, reuse=None):
    """

    :param input_tensor: [batch_size, M, H]
    :return:
    """
    net = layers.conv1d(input_tensor, config.spectral_hidden_dimension, kernel_size=7, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    # [2000 64]

    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    # [1000, 128], 0.15 params

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    # [500, 256], 0.6M params

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    # [250, 256], 0.6M

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, reuse=reuse)
    # [125, 256], 0.6M

    # finally a fully connected layer
    net = layers.conv1d(net, 1, kernel_size=config.M // 32, strides=1,
                        padding='valid', activation=None, kernel_regularizer=kernel_regularizer, reuse=reuse)
    logits = tf.squeeze(net, axis=[1, 2])

    return logits


def deep_match_scoring(aa_sequence, aa_sequence_length, ion_location_index, input_spectrum, keep_prob, reuse=None):
    with tf.variable_scope('embeddings', initializer=xavier_initializer(), reuse=reuse):
        amino_acid_embedding = tf.get_variable(name='aa_embedding',
                                               shape=[config.vocab_size, config.embed_dimension],
                                               # initializer=tf.random_uniform_initializer(minval=-1, maxval=1),
                                               trainable=True)

        position_embedding = tf.get_variable(name='position_embedding',
                                             shape=[config.peptide_max_length, config.embed_dimension],
                                             # initializer=tf.truncated_normal_initializer(stddev=0.2),
                                             trainable=True)
    mask_embedding = tf.concat([tf.zeros([1, 1]), tf.ones([config.vocab_size - 1, 1])],
                               axis=0, name='masked_emb')

    peptide_embedded = tf.nn.embedding_lookup(amino_acid_embedding, aa_sequence) + \
                       position_embedding
    peptide_pad_mask = tf.nn.embedding_lookup(mask_embedding, aa_sequence)
    peptide_embedded = peptide_embedded * peptide_pad_mask
    # dropout
    peptide_embedded = tf.nn.dropout(peptide_embedded, keep_prob=keep_prob, name='embedding_dropout')

    with tf.variable_scope('bidi_rnn', reuse=reuse):
        peptide, _, _ = bi_rnn(peptide_embedded, aa_sequence_length, config.lstm_output_dimension,
                               config.peptide_max_length, reuse=reuse)
        peptide_lstm_output = tf.stack(peptide, axis=1)
        # peptide: [batch_size, max_length, 2 * lstm_output_dimension]
    logger.info('peptide lstm output shape:')
    logger.info(peptide_lstm_output.get_shape())

    # explicitly bind lstm output
    temp_shape = tf.shape(peptide_lstm_output)
    zeros = tf.zeros((temp_shape[0], 1, temp_shape[2]), tf.float32)
    first_part = tf.concat((peptide_lstm_output, zeros), axis=1)
    second_part = tf.concat((zeros, peptide_lstm_output), axis=1)

    net = tf.concat((first_part, second_part), axis=2)
    net = net[:, 1:-1, :]

    with tf.variable_scope('spectral_transform', reuse=reuse):
        net = layers.dense(net, config.spectral_hidden_dimension, activation=tf.nn.relu,
                           kernel_regularizer=kernel_regularizer)
    logger.info('hidden output shape:')
    logger.info(net.get_shape())

    # get spectral representation
    location_shape = tf.shape(ion_location_index)
    resized_location = tf.reshape(ion_location_index, shape=(location_shape[0], location_shape[1] * location_shape[2]))
    resized_net = tf.reshape(net, shape=(location_shape[0], location_shape[1] * location_shape[2],
                                         config.spectral_hidden_dimension))
    theoretical_spectrum_shape = tf.stack([tf.cast(location_shape[0], tf.int64),
                                           config.M,
                                           config.spectral_hidden_dimension])
    theoretical_spectrum = batch_scatter(indices=resized_location, updates=resized_net,
                                         shape=theoretical_spectrum_shape, name='batch_scatter')

    logger.info('theoretical_spectrum shape:')
    logger.info(theoretical_spectrum.get_shape())

    # dropout on theoretical_spectrum
    theoretical_spectrum = tf.nn.dropout(theoretical_spectrum, keep_prob, name="theoretical_spectrum_dropout")
    combined_spectrum = tf.concat((theoretical_spectrum, input_spectrum), axis=-1)

    with tf.variable_scope('cnn_readout', reuse=reuse):
        output_logits = vgg_1d(combined_spectrum, reuse)
    return output_logits


class DeepMatchModel(object):
    def __init__(self, aa_sequence, aa_sequence_length, ion_location_index, input_spectrum):
        self.peptide_max_length = config.peptide_max_length
        self.embed_dimension = config.embed_dimension
        self.lstm_output_dimension = config.lstm_output_dimension
        self.vocab_size = config.vocab_size
        self.num_ion_combination = config.num_ion_combination
        self.spectral_hidden_dimension = config.spectral_hidden_dimension
        self.M = config.M

        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(config.weight_decay)

        ## do not use the old placeholder-feed_dict pattern for feeding data
        # self.aa_sequence = tf.placeholder(tf.int64, shape=(None, self.peptide_max_length),
        #                                               name='aa_sequence_placeholder')
        # self.aa_sequence_length = tf.placeholder(tf.int64, shape=(None,),
        #                                                      name='aa_sequence_length_placeholder')
        # self.ion_location_index = tf.placeholder(tf.int64,
        #                                                     shape=(None, self.peptide_max_length - 1,
        #                                                            self.num_ion_combination),
        #                                                     name="ion_location_index_placeholder")
        # self.input_spectrum = tf.placeholder(tf.float32, shape=(None, config.M, 1),
        #                                                  name='input_spectrum_placeholder')

        self.aa_sequence = aa_sequence
        self.aa_sequence_length = aa_sequence_length
        self.ion_location_index = ion_location_index
        self.input_spectrum = input_spectrum

        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

        with tf.variable_scope('embeddings'):
            self.amino_acid_embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dimension], -1, 1),
                                                    name='aa_embedding', trainable=True)
            self.position_embedding = tf.Variable(tf.truncated_normal([self.peptide_max_length,
                                                                       self.embed_dimension], 0, 0.2),
                                                  name='position_embedding', trainable=True)
        # this is masking out position embedding for _PAD tokens
        self.mask_embedding = tf.concat([tf.zeros([1, 1]), tf.ones([config.vocab_size - 1, 1])],
                                        axis=0, name='masked_emb')

        self._build()
        logging.info("total trainable params:")
        logging.info(f"{np.sum([np.prod(v.shape) for v in tf.trainable_variables()])}")

    def _vgg_1d(self, input_tensor):
        """

        :param input_tensor: [batch_size, M, H]
        :return:
        """
        net = layers.conv1d(input_tensor, self.spectral_hidden_dimension, kernel_size=7, strides=2,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        # [2000 64]

        net = layers.conv1d(net, 2 * self.spectral_hidden_dimension, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 2 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 2 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        # [1000, 128], 0.15 params

        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        # [500, 256], 0.6M params

        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        # [250, 256], 0.6M

        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=2,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        net = layers.conv1d(net, 4 * self.spectral_hidden_dimension, kernel_size=3, strides=1,
                            padding='same', activation=tf.nn.relu, kernel_regularizer=self.kernel_regularizer)
        # [125, 256], 0.6M

        # finally a fully connected layer
        net = layers.conv1d(net, 1, kernel_size=self.M // 32, strides=1,
                            padding='valid', activation=None, kernel_regularizer=self.kernel_regularizer)
        logits = tf.squeeze(net, axis=[1, 2])

        return logits

    def _build(self):
        peptide_embedded = tf.nn.embedding_lookup(self.amino_acid_embedding, self.aa_sequence) + \
                           self.position_embedding
        peptide_pad_mask = tf.nn.embedding_lookup(self.mask_embedding, self.aa_sequence)
        peptide_embedded = peptide_embedded * peptide_pad_mask
        # dropout
        peptide_embedded = tf.nn.dropout(peptide_embedded, keep_prob=self.keep_prob, name='embedding_dropout')

        with tf.variable_scope('bidi_rnn'):
            peptide, _, _ = bi_rnn(peptide_embedded, self.aa_sequence_length, self.lstm_output_dimension,
                                   self.peptide_max_length, reuse=False)
            peptide_lstm_output = tf.stack(peptide, axis=1)
            # peptide: [batch_size, max_length, 2 * lstm_output_dimension]
        logger.info('peptide lstm output shape:')
        logger.info(peptide_lstm_output.get_shape())

        # explicitly bind lstm output
        temp_shape = tf.shape(peptide_lstm_output)
        zeros = tf.zeros((temp_shape[0], 1, temp_shape[2]), tf.float32)
        first_part = tf.concat((peptide_lstm_output, zeros), axis=1)
        second_part = tf.concat((zeros, peptide_lstm_output), axis=1)

        net = tf.concat((first_part, second_part), axis=2)
        net = net[:, 1:-1, :]

        with tf.variable_scope('spectral_transform'):
            net = layers.dense(net, self.spectral_hidden_dimension, activation=tf.nn.relu,
                               kernel_regularizer=self.kernel_regularizer)
        logger.info('hidden output shape:')
        logger.info(net.get_shape())

        # get spectral representation
        location_shape = tf.shape(self.ion_location_index)
        resized_location = tf.reshape(self.ion_location_index, shape=(location_shape[0],
                                                                      location_shape[1] * location_shape[2]))
        resized_net = tf.reshape(net, shape=(location_shape[0], location_shape[1] * location_shape[2],
                                             self.spectral_hidden_dimension))
        theoretical_spectrum_shape = tf.stack([tf.cast(location_shape[0], tf.int64),
                                               self.M,
                                               self.spectral_hidden_dimension])
        theoretical_spectrum = batch_scatter(indices=resized_location, updates=resized_net,
                                             shape=theoretical_spectrum_shape, name='batch_scatter')

        logger.info('theoretical_spectrum shape:')
        logger.info(theoretical_spectrum.get_shape())

        # dropout on theoretical_spectrum
        theoretical_spectrum = tf.nn.dropout(theoretical_spectrum, self.keep_prob, name="theoretical_spectrum_dropout")
        combined_spectrum = tf.concat((theoretical_spectrum, self.input_spectrum), axis=-1)

        with tf.variable_scope('cnn_readout'):
            self.output_logits = self._vgg_1d(combined_spectrum)
