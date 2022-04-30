from collections import namedtuple
from functools import reduce
from math import sqrt

import numpy as np

Decomposition = namedtuple('Decomposition', ['Q', 'R'])


def QR_decomposition(A: np.ndarray) -> Decomposition:
    n = A.shape[0]

    Q, A_k = np.eye(n), A.copy()
    for i in range(n):
        for j in range(i + 1, n):
            c, s = A_k[i, i] / sqrt(A_k[i, i] ** 2 + A_k[j, i] ** 2), -A_k[j, i] / sqrt(A_k[i, i] ** 2 + A_k[j, i] ** 2)
            T = np.array([[c if row == column == i or row == column == j else
                           -s if row == i and column == j else s if row == j and column == i else
                           1 if row == column else 0 for column in range(n)] for row in range(n)])
            Q, A_k = Q.dot(T.transpose()), T.dot(A_k)

    # A = Q * R => R = Q^(-1) * A
    return Decomposition(Q, Q.transpose().dot(A))


# |A| != 0
def solve_using_QR_decomposition(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    def solve_Rx_v(R, v):
        return np.array(reduce(lambda x, row: [(v[row] - sum([R[row, column] * x[column - row - 1]
                                                              for column in range(row + 1, n)])) / R[row, row]] + x,
                               range(n - 1, -1, -1), []))

    if not np.linalg.det(A):
        raise ValueError('|A| = 0.')

    n = A.shape[0]

    decomposition = QR_decomposition(A)

    return solve_Rx_v(decomposition.R, decomposition.Q.transpose().dot(b))
