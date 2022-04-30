import unittest

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task2.task2a import LU_decomposition, solve_using_LU_decomposition


class TestTask2a(unittest.TestCase):
    def test_LU_decomposition_check_if_L_and_U_are_triangular(self):
        A = np.random.rand(100, 100)
        L, U = LU_decomposition(A)

        for row in range(100):
            for column in range(100):
                self.assertTrue(row < column and L[row, column] == 0 or
                                row == column and L[row, column] == 1 or
                                row > column and U[row, column] == 0)

    def test_LU_decomposition_with_diagonal_matrix_with_dimension_4(self):
        A = np.array([[17, 0, 0, 0], [0, 4, 0, 0], [0, 0, -93, 0], [0, 0, 0, 5]])
        L, U = LU_decomposition(A)

        self.assertTrue(np.array_equal(A, L.dot(U)))

    def test_LU_decomposition_with_hilberts_matrix_with_dimension_5(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(5)] for row in range(5)])
        L, U = LU_decomposition(A)

        self.assertTrue(np.allclose(A, L.dot(U)))

    def test_LU_decomposition_with_tridiagonal_matrix_with_dimension_5(self):
        A = np.array([[generate_tridiagonal_matrix_element(row, column) for column in range(5)] for row in range(5)])
        L, U = LU_decomposition(A)

        self.assertTrue(np.allclose(A, L.dot(U)))

    def test_solve_using_LU_decomposition_with_null_determinant(self):
        A = np.array([[0, 0, 0], [1, -4, 8], [-71, 0, 13]])
        b = np.array([1, 3, 1])

        self.assertRaises(ValueError, lambda: solve_using_LU_decomposition(A, b))

    def test_solve_using_LU_decomposition_with_diagonal_matrix_with_dimension_4(self):
        A = np.array([[25, 0, 0, 0], [0, -4, 0, 0], [0, 0, 16, 0], [0, 0, 0, 800]])
        x = np.ones(4)
        b = A.dot(x)

        self.assertTrue(np.array_equal(x, solve_using_LU_decomposition(A, b)))

    def test_solve_using_LU_decomposition_with_hilberts_matrix_with_dimension_5(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(5)] for row in range(5)])
        x = np.ones(5)
        b = A.dot(x)

        self.assertTrue(np.allclose(x, solve_using_LU_decomposition(A, b)))

    def test_solve_using_LU_decomposition_with_tridiagonal_matrix_with_dimension_10(self):
        A = np.array([[generate_tridiagonal_matrix_element(row, column) for column in range(10)] for row in range(10)])
        x = np.ones(10)
        b = A.dot(x)

        self.assertTrue(np.allclose(x, solve_using_LU_decomposition(A, b)))

    def test_solve_using_LU_decomposition_with_random_matrix_with_dimension_100(self):
        A = np.random.rand(100, 100)
        x = np.ones(100)
        b = A.dot(x)

        self.assertTrue(np.allclose(x, solve_using_LU_decomposition(A, b)))
