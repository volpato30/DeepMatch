import unittest
import config
import cython_func


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


if __name__ == '__main__':
    unittest.main()
