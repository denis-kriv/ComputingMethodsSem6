from collections import namedtuple
from functools import reduce

import numpy as np

Decomposition = namedtuple('Decomposition', ['L', 'U'])


def LU_decomposition(A: np.ndarray) -> Decomposition:
    n = A.shape[0]

    L, U = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            L_koef, U_koef = reduce(lambda p, k: (p[0] + L[j, k] * U[k, i], p[1] + L[i, k] * U[k, j]), range(i), (0, 0))
            U[i, j] = A[i, j] - U_koef
            L[j, i] = (A[j, i] - L_koef) / U[i, i]

    return Decomposition(L, U)


# |A| != 0
def solve_using_LU_decomposition(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    def solve_Lx_v(L, v):
        return np.array(reduce(lambda x, row: x + [(v[row] - sum([L[row, column] * x[column]
                                                                  for column in range(row)])) / L[row, row]],
                               range(n), []))

    def solve_Ux_v(U, v):
        return np.array(reduce(lambda x, row: [(v[row] - sum([U[row, column] * x[column - row - 1]
                                                              for column in range(row + 1, n)])) / U[row, row]] + x,
                               range(n - 1, -1, -1), []))

    if not np.linalg.det(A):
        raise ValueError('det(A) = 0')

    n = A.shape[0]

    decomposition = LU_decomposition(A)

    return solve_Ux_v(decomposition.U, solve_Lx_v(decomposition.L, b))
