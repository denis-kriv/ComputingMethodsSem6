from itertools import product

import numpy as np

from practical_tasks.solutions.task4.common import solve_using_iterative_method, SolutionUsingIterativeMethod


# A with diagonal dominance
def solve_using_zeidels_method(A: np.ndarray, b: np.ndarray, epsilon: float) -> SolutionUsingIterativeMethod:
    n = A.shape[0]

    if any([2 * abs(A[row, row]) <= sum([abs(A[row, column]) for column in range(n)]) for row in range(n)]):
        raise ValueError('A without diagonal dominance.')

    L, R = np.zeros((n, n)), np.zeros((n, n))
    for row, column in product(range(n), range(n)):
        L[row, column], R[row, column] = (A[row, column], 0) if row >= column else (0, A[row, column])
    L_inverse = np.linalg.inv(L)
    B, c = -L_inverse.dot(R), L_inverse.dot(b)

    return solve_using_iterative_method(B, c, epsilon)
