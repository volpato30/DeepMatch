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
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv11",
                        reuse=reuse)
    net = layers.conv1d(net, config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv12",
                        reuse=reuse)
    net = layers.conv1d(net, config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv13",
                        reuse=reuse)
    # [2000 64]

    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv21",
                        reuse=reuse)
    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv22",
                        reuse=reuse)
    net = layers.conv1d(net, 2 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv23",
                        reuse=reuse)
    # [1000, 128], 0.15 params

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv31",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv32",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv33",
                        reuse=reuse)
    # [500, 256], 0.6M params

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv41",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv42",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv43",
                        reuse=reuse)
    # [250, 256], 0.6M

    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=2,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv51",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv52",
                        reuse=reuse)
    net = layers.conv1d(net, 4 * config.spectral_hidden_dimension, kernel_size=3, strides=1,
                        padding='same', activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, name="conv53",
                        reuse=reuse)
    # [125, 256], 0.6M

    # finally a fully connected layer
    net = layers.conv1d(net, 1, kernel_size=config.M // 32, strides=1,
                        padding='valid', activation=None, kernel_regularizer=kernel_regularizer, name="conv_final",
                        reuse=reuse)
    logits = tf.squeeze(net, axis=[1, 2])

    return logits


def deep_match_scoring(aa_sequence, aa_sequence_length, ion_location_index, input_spectrum, keep_prob, reuse=None):
    """
    build computation graph
    :param aa_sequence: [batch, peptide_max_length]
    :param aa_sequence_length: [batch, 1]
    :param ion_location_index: notice the index can be negative or out of bound, need to process in the model
    :param input_spectrum:
    :param keep_prob:
    :param reuse:
    :return:
    """
    if aa_sequence_length.get_shape().ndims == 2:
        aa_sequence_length = tf.squeeze(aa_sequence_length, 1)
    if input_spectrum.get_shape().ndims == 2:
        input_spectrum = tf.expand_dims(input_spectrum, axis=2)
    assert aa_sequence_length.get_shape().ndims == 1
    assert input_spectrum.get_shape().ndims == 3
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
                       tf.expand_dims(position_embedding, axis=0)
    peptide_pad_mask = tf.nn.embedding_lookup(mask_embedding, aa_sequence)
    peptide_embedded = peptide_embedded * peptide_pad_mask
    logger.info('peptide_embedded shape:')
    logger.info(peptide_embedded.get_shape())

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
        net = layers.dense(net, config.num_ion_combination * config.spectral_hidden_dimension, activation=tf.nn.relu,
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

    # Mask out Out Of Range location indices
    mask_matrix = tf.cast(resized_location < config.M, tf.int64)
    resized_location = resized_location * mask_matrix

    float_mask_matrix = tf.expand_dims(tf.cast(mask_matrix, tf.float32), axis=2)
    resized_net = resized_net * float_mask_matrix

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
