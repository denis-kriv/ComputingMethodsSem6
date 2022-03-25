from collections import namedtuple
from functools import reduce

import numpy as np
from scipy.linalg import solve

from common import create_table, create_plot

TableRow = namedtuple('TableRow', ['n', 'cond_A', 'cond_B', 'error_norm'])
ExperimentResult = namedtuple('ExperimentResult', ['alpha', 'fst_error_norm', 'snd_error_norm'])


# A -- generalized Vandermorde`s matrix
def check_if_oscillation_type(A: np.ndarray) -> bool:
    n = A.shape[0]
    return np.linalg.det(A) > 0 and all([(i < 1 or A[i][i - 1] > 0) and A[i][i] > 0 and (i > n - 2 or A[i][i + 1] > 0)
                                         for i in range(n)])


# using Newton`s method
def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return reduce(lambda m, _: (m + np.linalg.inv(m).dot(A)) / 2, range(n + 1), A)


# (conjugate(A) * A + alpha * E) * x = conjugate(A) * b
def solve_using_first_regularization_method(A: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    n = A.shape[0]
    conjugate_A = np.matrix.conjugate(A)
    return solve(conjugate_A.dot(A) + alpha * np.identity(n), conjugate_A.dot(b))


# B^2 = A
# (conjugate(B) * B + alpha * E) * x = conjugate(B) * (inv(B) * b)
def solve_using_second_regularization_method(B: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    n = B.shape[0]
    conjugate_B = np.matrix.conjugate(B)
    return solve(conjugate_B.dot(B) + alpha * np.identity(n), conjugate_B.dot(np.linalg.inv(B).dot(b)))


def run(generate_matrix_element, epsilon, alpha_range):
    table = []
    results = {}
    n = 2
    difference_norm = 0
    while difference_norm < epsilon:
        A = np.array([[generate_matrix_element(n, row, column) for column in range(n)] for row in range(n)])
        x = np.ones(n)
        b = A.dot(x)

        if not check_if_oscillation_type(A):
            break

        B = matrix_sqrt(A)
        difference_norm = np.linalg.norm(A - B.dot(B))
        table.append(TableRow(n, np.linalg.cond(A), np.linalg.cond(B), difference_norm))

        if difference_norm < epsilon:
            results[n] = [(alpha, np.linalg.norm(x - solve_using_first_regularization_method(A, b, alpha)),
                           np.linalg.norm(x - solve_using_second_regularization_method(A, b, alpha)))
                          for alpha in alpha_range]
        n += 1

    create_table('results', 'finding_sqrt_A_results.csv', TableRow._fields, table)

    for k in results:
        _, fst_error_norms, snd_error_norms = zip(*results[k])
        create_plot('results', f'result with n={k}.png', alpha_range,
                    [(list(fst_error_norms), '||x - x_fst||'), (list(snd_error_norms), '||x - x_snd||')],
                    'n = 5', 'alpha', 'error_norm', 'upper right')


if __name__ == '__main__':
    run(lambda dimension, row, column: 1 / (dimension - row) ** (2 * (column + 1)),
        10**(-1), [10 ** (-i) for i in range(1, 4)])
