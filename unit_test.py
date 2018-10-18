import unittest
import logging
import logging.config
import tensorflow as tf
import numpy as np
import config
import cython_func
from InputParser import TrainParser, make_dataset
from utils import batch_scatter_add, batch_scatter

logger = logging.getLogger(__name__)


class TestCythonFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_ions_location_index(self):
        peptide_mass = 1000.
        prefix_mass = 100.

        # b, a, y
        assert config.delta_M == 0.5
        # the expected_result is computed under the assumption that delta_M is 0.5
        expected_result = [202, 165, 167, 146, 110, 111, 1802, 1765, 1767]
        result_index = cython_func.get_ions_mz_index(peptide_mass, prefix_mass)
        for i, target in enumerate(expected_result):
            self.assertEqual(result_index[i], target, msg=f"left: {result_index[i]} not equal to "
                                                          f"right: {target}")

    def test_process_spectrum(self):
        assert config.delta_M == 0.5
        mz_list = [103.2, 209.5]
        intensity_list = [100.0, 1000.0]

        spectrum_holder = cython_func.process_spectrum(mz_list, intensity_list)
        spectrum_holder_expected = np.zeros(config.M)
        spectrum_holder_expected[206] = 1. / 11
        spectrum_holder_expected[419] = 10. / 11

        self.assert_(np.allclose(spectrum_holder_expected, spectrum_holder))

    def test_config_aa_mass(self):
        self.assertAlmostEqual(config.mass_ID[config.vocab['K']], 128.09496)


class TestBatchScatterAdd(unittest.TestCase):
    def setUp(self):
        batch_size = 2
        num_ions = 2
        hidden_size = 3
        M = 5
        self.hidden_repre_np = np.arange(batch_size * 1 * num_ions * hidden_size).reshape(
            batch_size, 1, num_ions, hidden_size).astype(np.float32)

        self.location_np = np.array([[0, 4], [2, 4]]).astype(np.int32).reshape(2, 1, 2)
        self.location_index = tf.placeholder(tf.int32, (None, 1, num_ions), name='location')
        ion_hidden_represent = tf.Variable(self.hidden_repre_np, name='hidden_represent', trainable=True)
        resized_ion_hidden = tf.reshape(ion_hidden_represent, (-1, num_ions, hidden_size))
        resized_location_index = tf.reshape(self.location_index, (-1, num_ions))

        output = tf.zeros((batch_size, M, hidden_size), dtype=tf.float32)

        update = resized_ion_hidden
        self.output = batch_scatter_add(ref=output, indices=resized_location_index, updates=update,
                                        name='batch_scatter_add')
        self.final_output = tf.reduce_mean(self.output)
        self.gradient_to_hidden = tf.gradients(self.final_output, ion_hidden_represent)[0]

    def test_forward_computation(self):
        with tf.Session() as sess:
            init_op = tf.initializers.variables(tf.global_variables())
            sess.run(init_op)
            # print(sess.run(gathered_output, feed_dict={location_index:location_np}))
            r1 = sess.run(self.final_output, feed_dict={self.location_index: self.location_np})
            r2 = sess.run(self.final_output, feed_dict={self.location_index: self.location_np})
            out = sess.run(self.output, feed_dict={self.location_index: self.location_np})

            self.assertAlmostEqual(r1, np.sum(np.arange(2*2*3)) / (2 * 5 *3))
            self.assertAlmostEqual(r2, np.sum(np.arange(2*2*3)) / (2 * 5 *3))
            out_true = np.zeros((2, 5, 3), np.float32)
            out_true[0, 0] = np.array([0., 1., 2.])
            out_true[0, 4] = np.array([3., 4., 5.])
            out_true[1, 2] = np.array([6., 7., 8.])
            out_true[1, 4] = np.array([9., 10., 11.])

            self.assert_(np.allclose(out, out_true))

    def test_backward_computation(self):
        with tf.Session() as sess:
            init_op = tf.initializers.variables(tf.global_variables())
            sess.run(init_op)
            g = sess.run(self.gradient_to_hidden, feed_dict={self.location_index: self.location_np})
            g_true = np.ones((2, 1, 2, 3), np.float32) / (2 * 5 * 3)
            self.assert_(np.allclose(g, g_true))


class TestBatchScatter(unittest.TestCase):
    def setUp(self):
        batch_size = 2
        num_ions = 2
        hidden_size = 3
        M = 5
        self.hidden_repre_np = np.arange(batch_size * 1 * num_ions * hidden_size).reshape(
            batch_size, 1, num_ions, hidden_size).astype(np.float32)
        # duplicate in indices, and also include an invalid index(greater than M-1)
        self.location_np = np.array([[0, 0], [2, 5]]).astype(np.int64).reshape(2, 1, 2)
        self.location_index = tf.placeholder(tf.int64, (None, 1, num_ions), name='location')
        ion_hidden_represent = tf.Variable(self.hidden_repre_np, name='hidden_represent', trainable=True)
        resized_ion_hidden = tf.reshape(ion_hidden_represent, (-1, num_ions, hidden_size))
        resized_location_index = tf.reshape(self.location_index, (-1, num_ions))

        update = resized_ion_hidden
        shape = tf.cast(tf.stack([tf.shape(self.location_index)[0], M, hidden_size]), tf.int64)
        # if a position in the location_index is greater than M-1 then mask it out
        mask_matrix = tf.cast(resized_location_index < M, tf.int64)
        resized_location_index = resized_location_index * mask_matrix

        mask_matrix = tf.expand_dims(tf.cast(mask_matrix, tf.float32), axis=2)
        print(f"float mask matrix shape: {mask_matrix.get_shape()}")
        update = mask_matrix * update
        print(f"float update shape: {update.get_shape()}")
        self.output = batch_scatter(indices=resized_location_index, updates=update, shape=shape, name='batch_scatter')

        self.final_output = tf.reduce_mean(self.output)
        self.gradient_to_hidden = tf.gradients(self.final_output, ion_hidden_represent)[0]

    def test_forward_computation(self):
        with tf.Session() as sess:
            init_op = tf.initializers.variables(tf.global_variables())
            sess.run(init_op)
            # print(sess.run(gathered_output, feed_dict={location_index:location_np}))
            r1 = sess.run(self.final_output, feed_dict={self.location_index: self.location_np})
            r2 = sess.run(self.final_output, feed_dict={self.location_index: self.location_np})
            out = sess.run(self.output, feed_dict={self.location_index: self.location_np})

            self.assertAlmostEqual(r1, np.sum(np.arange(2*2*3 - 3)) / (2 * 5 *3))
            self.assertAlmostEqual(r2, np.sum(np.arange(2*2*3 - 3)) / (2 * 5 *3))
            out_true = np.zeros((2, 5, 3), np.float32)
            out_true[0, 0] = np.array([0., 1., 2.]) + np.array([3., 4., 5.])
            out_true[1, 2] = np.array([6., 7., 8.])

            self.assert_(np.allclose(out, out_true))

    def test_backward_computation(self):
        with tf.Session() as sess:
            init_op = tf.initializers.variables(tf.global_variables())
            sess.run(init_op)
            g = sess.run(self.gradient_to_hidden, feed_dict={self.location_index: self.location_np})
            g_true = np.ones((2, 1, 2, 3), np.float32) / (2 * 5 * 3)
            g_true[1, 0, 1] = 0
            print(f"gradient shape: {g.shape}")
            self.assertTrue(np.allclose(g, g_true), msg=f"g:\n{g}")


class TestReader(unittest.TestCase):
    def setUp(self):
        self.train_parser = TrainParser(spectrum_file="./test_data/test_scans.txt")

    def test_convert_to_tfrecord(self):
        self.train_parser.convert_to_tfrecord("./test_data/test_scans.tfrecord", test_mode=True)

    def test_read_tfrecord(self):
        """
        Integration test
        :return:
        """
        dataset = make_dataset("./test_data/test_scans.tfrecord", batch_size=1, num_processes=1)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
        init_op = iterator.make_initializer(dataset)

        with tf.Session() as sess:
            sess.run(init_op)
            first_dp = sess.run(next_element)
            logger.debug(f"pos_aa_sequence_length have shape: {first_dp['pos_aa_sequence_length'].shape}")
            # check length
            self.assertEqual(first_dp['pos_aa_sequence_length'][0][0], 8)
            self.assertEqual(first_dp['neg_aa_sequence_length'][0][0], 9)

            pos_peptide = list("KKIYEEKK")
            pos_peptide = [config.vocab[x] for x in pos_peptide]
            pos_peptide = TrainParser.pad_to_length(pos_peptide, config.peptide_max_length, config.PAD_ID)

            neg_peptide = list("KGQKRSFSK")
            neg_peptide = [config.vocab[x] for x in neg_peptide]
            neg_peptide = TrainParser.pad_to_length(neg_peptide, config.peptide_max_length, config.PAD_ID)

            self.assertListEqual(pos_peptide, first_dp['pos_aa_sequence'][0].tolist())
            self.assertListEqual(neg_peptide, first_dp['neg_aa_sequence'][0].tolist())

            self.assertAlmostEqual(np.sum(first_dp['input_spectrum']), 1.0)
            self.assertGreater(first_dp["input_spectrum"][0][784], 0.0)
            self.assertGreater(first_dp["input_spectrum"][0][1839], 0.0)
            self.assertGreater(first_dp["input_spectrum"][0][160], 0.0)

            for i in first_dp["pos_ion_location_index"][0][-1].tolist():
                # pad pos_ion_location with config.M
                self.assertEqual(i, config.M)

            for i in first_dp["neg_ion_location_index"][0][-1].tolist():
                # pad pos_ion_location with config.M
                self.assertEqual(i, config.M)

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
    unittest.main()
