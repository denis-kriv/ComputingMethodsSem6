import random
import unittest

import numpy as np

from practical_tasks.solutions.task4.task4b import solve_using_zeidels_method


class TestTask4b(unittest.TestCase):
    def test_solve_using_zeidels_method_with_incorrect_matrix_A_should_throws_value_error(self):
        A = np.array([[10, 14, -50, 0], [4, 1, 2, -1], [13, 6030, -1000, 1], [0, 0, -72, 3]])
        x = np.ones(4)
        b = A.dot(x)
        epsilon = 10 ** (-8)

        self.assertRaises(ValueError, lambda: solve_using_zeidels_method(A, b, epsilon))

    def test_solve_using_zeidels_method_with_correct_matrix_A_should_works_correctly(self):
        A = np.array([[random.randint(100, 1000) if row == column else random.randint(-1, 9)
                       for column in range(10)] for row in range(10)])
        x = np.ones(10)
        b = A.dot(x)
        epsilon = 10 ** (-8)

        self.assertTrue(np.allclose(x, solve_using_zeidels_method(A, b, epsilon).x))
