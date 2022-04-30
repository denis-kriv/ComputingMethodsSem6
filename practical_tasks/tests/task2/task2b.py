import unittest

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task2.task2b import solve_using_regularization


class TestTask2b(unittest.TestCase):
    def test_solve_using_regularization_with_null_determinant(self):
        A = np.array([[0, 0, 0], [1, -4, 8], [-71, 0, 13]])
        b = np.array([1, 1, 1])

        self.assertRaises(ValueError, lambda: solve_using_regularization(A, b, 10**(-12)))

    def test_solve_using_regularization_with_hilberts_matrix_with_dimension_5(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(5)] for row in range(5)])
        x = np.ones(5)
        b = A.dot(x)

        for alpha in [10**(-i) for i in range(15, 20)]:
            sol = solve_using_regularization(A, b, alpha).x
            print(np.linalg.norm(x - sol))
            self.assertTrue(np.allclose(x, solve_using_regularization(A, b, alpha).x))

    def test_solve_using_regularization_with_tridiagonal_matrix_with_dimension_10(self):
        A = np.array([[generate_tridiagonal_matrix_element(row, column) for column in range(10)] for row in range(10)])
        x = np.ones(10)
        b = A.dot(x)

        for alpha in [10 ** (-i) for i in range(15, 20)]:
            self.assertTrue(np.allclose(x, solve_using_regularization(A, b, alpha).x))
