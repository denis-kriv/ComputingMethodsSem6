import unittest
from math import inf

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element
from practical_tasks.solutions.task1.task1 import ConditionNumbers, calculate_condition_numbers


class TestTask1(unittest.TestCase):
    def test_calculate_condition_numbers_with_null_determinant(self):
        A = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [12, 27, 93, 0], [181, -59, 18, 0]])

        self.assertEqual(ConditionNumbers(inf, inf, inf), calculate_condition_numbers(A))

    def test_calculate_condition_numbers_with_matrix_with_dimension_3(self):
        A = np.array([[17, 27, 11], [34, -5, 0], [-8, 872, -1]])

        self.assertTrue(sum(calculate_condition_numbers(A)) < 100)

    def test_calculate_condition_numbers_with_matrix_with_dimension_1(self):
        A = np.array([[2]])

        self.assertEqual(ConditionNumbers(1, 1, 1), calculate_condition_numbers(A))

    def test_calculate_condition_numbers_with_hilberts_matrix_with_dimension_20(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(20)] for row in range(20)])

        self.assertGreater(calculate_condition_numbers(A).spectral, 10**4)
