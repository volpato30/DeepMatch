import unittest
import tensorflow as tf
import numpy as np
import config
import cython_func
from utils import batch_scatter_add


class TestCythonFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_ions_location_index(self):
        peptide_mass = 1000
        prefix_mass = 100

        # b, a, y
        assert config.delta_M == 0.5
        # the expected_result is computed under the assumpt that delta_M is 0.5
        expected_result = [202, 165, 167, 146, 110, 111, 1802, 1765, 1767]
        result_index = cython_func.get_ions_mz_index(1000, 100)
        for i, target in enumerate(expected_result):
            self.assertEqual(result_index[i], target, msg=f"left: {result_index[i]} not equal to "
                                                          f"right: {target}")


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

        gathered_output = tf.batch_gather(params=output,
                                          indices=resized_location_index)  # (batch, resize_location_index, hidden_size)
        update = resized_ion_hidden
        self.output = batch_scatter_add(ref=output, indices=resized_location_index, updates=update, name='batch_scatter_add')
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
            g_true = np.ones((2, 2, 3), np.float32) / (2 * 5 * 3)
            self.assert_(np.allclose(g, g_true))


if __name__ == '__main__':
    unittest.main()
