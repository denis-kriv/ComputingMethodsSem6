import unittest

import numpy as np

from practical_tasks.solutions.task6.task6a import find_eigen_values_using_first_method


class TestTask6a(unittest.TestCase):
    def test_find_eigen_values_using_first_method_with_incorrect_matrix_should_throws_value_error(self):
        A = np.random.rand(10, 10)
        epsilon = 10 ** (-8)
        max_iterations_count = 10000

        self.assertRaises(ValueError, lambda: find_eigen_values_using_first_method(A, epsilon, max_iterations_count))

    def test_find_eigen_values_using_first_method_with_many_iterations_should_throws_value_error(self):
        A = np.array([[1, 3, 0], [3, 2, 6], [0, 6, 5]])
        epsilon = 10 ** (-8)
        max_iterations_count = 3

        self.assertRaises(ValueError, lambda: find_eigen_values_using_first_method(A, epsilon, max_iterations_count))

    def test_find_eigen_values_using_first_method_with_correct_data_should_works_correctly(self):
        A = np.array([[1, 3, 0], [3, 2, 6], [0, 6, 5]])
        epsilon = 10 ** (-8)
        max_iterations_count = 1000000

        self.assertTrue(np.allclose(sorted(np.linalg.eigvals(A)),
                                    sorted(find_eigen_values_using_first_method(A, epsilon, max_iterations_count)[0])))
