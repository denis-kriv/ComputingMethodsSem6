import unittest

import numpy as np

from practical_tasks.solutions.task5.task5b import find_using_scalar_product_method


class TestTask5b(unittest.TestCase):
    def test_find_using_scalar_product_method_with_incorrect_matrix_should_throws_value_error(self):
        A = np.random.rand(10, 10)
        epsilon = 10 ** (-8)
        max_iterations_count = 10000

        self.assertRaises(ValueError, lambda: find_using_scalar_product_method(A, epsilon, max_iterations_count))

    def test_find_using_scalar_product_method_with_many_iterations_should_throws_value_error(self):
        A = np.array([[1, 3, 0], [3, 2, 6], [0, 6, 5]])
        epsilon = 10 ** (-8)
        max_iterations_count = 3

        self.assertRaises(ValueError, lambda: find_using_scalar_product_method(A, epsilon, max_iterations_count))

    def test_find_using_scalar_product_method_with_correct_data_should_works_correctly(self):
        A = np.array([[1, 3, 0], [3, 2, 6], [0, 6, 5]])
        epsilon = 10 ** (-8)
        max_iterations_count = 10000

        self.assertAlmostEqual(sorted(np.linalg.eigvals(A), key=abs)[-1],
                               find_using_scalar_product_method(A, epsilon, max_iterations_count).eigen_value)
