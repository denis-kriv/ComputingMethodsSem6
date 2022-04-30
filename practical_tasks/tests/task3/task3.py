import unittest

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task3.task3 import QR_decomposition, solve_using_QR_decomposition


class TestTask3(unittest.TestCase):
    def test_QR_decomposition_check_if_Q_is_ortogonal_and_R_is_triangular(self):
        A = np.random.rand(30, 30)
        Q, R = QR_decomposition(A)

        self.assertTrue(np.allclose(np.linalg.inv(Q), Q.transpose()) and
                        all([[row < column or R[row, column] == 0 for column in range(30)] for row in range(30)]))

    def test_QR_decomposition_with_diagonal_matrix_with_dimension_4(self):
        A = np.array([[17, 0, 0, 0], [0, 4, 0, 0], [0, 0, -93, 0], [0, 0, 0, 5]])
        Q, R = QR_decomposition(A)

        self.assertTrue(np.array_equal(A, Q.dot(R)))

    def test_QR_decomposition_with_hilberts_matrix_with_dimension_5(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(5)] for row in range(5)])
        Q, R = QR_decomposition(A)

        self.assertTrue(np.allclose(A, Q.dot(R)))

    def test_QR_decomposition_with_tridiagonal_matrix_with_dimension_5(self):
        A = np.array([[generate_tridiagonal_matrix_element(row, column) for column in range(5)] for row in range(5)])
        Q, R = QR_decomposition(A)

        self.assertTrue(np.allclose(A, Q.dot(R)))

    def test_solve_using_QR_decomposition_with_null_determinant(self):
        A = np.array([[0, 0, 0], [1, -4, 8], [-71, 0, 13]])
        b = np.array([1, 1, 1])

        self.assertRaises(ValueError, lambda: solve_using_QR_decomposition(A, b))

    def test_solve_using_QR_decomposition_with_diagonal_matrix_with_dimension_4(self):
        A = np.array([[25, 0, 0, 0], [0, -4, 0, 0], [0, 0, 16, 0], [0, 0, 0, 800]])
        x = np.ones(4)
        b = A.dot(x)

        self.assertTrue(np.array_equal(x, solve_using_QR_decomposition(A, b)))

    def test_solve_using_QR_decomposition_with_hilberts_matrix_with_dimension_5(self):
        A = np.array([[generate_hilberts_matrix_element(row, column) for column in range(5)] for row in range(5)])
        x = np.ones(5)
        b = A.dot(x)

        print(solve_using_QR_decomposition(A, b))

        self.assertTrue(np.allclose(x, solve_using_QR_decomposition(A, b)))

    def test_solve_using_QR_decomposition_with_tridiagonal_matrix_with_dimension_10(self):
        A = np.array([[generate_tridiagonal_matrix_element(row, column) for column in range(10)] for row in range(10)])
        x = np.ones(10)
        b = A.dot(x)

        self.assertTrue(np.allclose(x, solve_using_QR_decomposition(A, b)))

    def test_solve_using_QR_decomposition_with_random_matrix_with_dimension_100(self):
        A = np.random.rand(40, 40)
        x = np.ones(40)
        b = A.dot(x)

        self.assertTrue(np.allclose(x, solve_using_QR_decomposition(A, b)))
