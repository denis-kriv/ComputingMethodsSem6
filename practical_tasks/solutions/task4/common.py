from collections import namedtuple

import numpy as np

SolutionUsingIterativeMethod = namedtuple('SolutionUsingIterativeMethod', ['x', 'iterations_count'])


# p(B) < 1
def solve_using_iterative_method(B: np.ndarray, c: np.ndarray, epsilon: float) -> SolutionUsingIterativeMethod:
    n = B.shape[0]

    if max([abs(v) for v in np.linalg.eigvals(B)]) >= 1:
        raise ValueError('p(B) >= 1')

    B_norm = np.linalg.norm(B)
    t = B_norm / (1 - B_norm)
    x, iterations_count = np.zeros(n), 0
    while True:
        previous_x = x
        x = B.dot(previous_x) + c
        iterations_count += 1
        # a posteriori estimate: ||x_k - x|| <= t * ||x_k - x_{k - 1}||
        if t * np.linalg.norm(x - previous_x) < epsilon:
            break

    return SolutionUsingIterativeMethod(x, iterations_count)
