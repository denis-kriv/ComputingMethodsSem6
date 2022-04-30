from collections import namedtuple
from math import sqrt
from typing import Callable

import numpy as np

EigenValues = namedtuple('EigenValues', ['eigen_values', 'iterations_count'])


# A = A*
def find_eigen_values(A: np.ndarray, epsilon: float, max_iterations_count: int, get_element: Callable) -> EigenValues:
    n = A.shape[0]

    if not np.allclose(A.T, np.conjugate(A)):
        raise ValueError('A != A*')

    A_k, iterations_count, i, j = A.copy(), 0, 0, 0
    while True:
        i, j = get_element(A_k, i, j)
        x, y = -2 * A_k[i, j], A_k[i, i] - A_k[j, j]
        if y == 0:
            c = s = 1 / sqrt(2)
        else:
            c = sqrt((1 + abs(y) / sqrt(x ** 2 + y ** 2)) / 2)
            s = (-1 if x * y < 0 else 1 if x * y > 0 else 0) * abs(x) / (2 * c * sqrt(x ** 2 + y ** 2))
        T = np.array([[c if row == column == i or row == column == j else
                       -s if row == i and column == j else
                       s if row == j and column == i else
                       1 if row == column else 0 for column in range(n)] for row in range(n)])

        A_k = T.dot(A_k).dot(T.transpose())
        iterations_count += 1

        # check if R_i < epsilon for all i
        if all([sum([abs(A_k[row, column]) for column in range(n)]) - abs(A_k[row, row]) < epsilon
                for row in range(n)]):
            break

        if iterations_count > max_iterations_count:
            raise ValueError(f'Result not found after {max_iterations_count} iterations.')

    return EigenValues([A_k[row, row] for row in range(n)], iterations_count)
