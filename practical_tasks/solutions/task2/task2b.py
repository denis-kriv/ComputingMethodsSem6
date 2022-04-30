from collections import namedtuple

import numpy as np

from practical_tasks.solutions.task1.task1 import calculate_condition_numbers
from practical_tasks.solutions.task2.task2a import solve_using_LU_decomposition

SolutionUsingRegularization = namedtuple('SolutionUsingRegularization', ['x', 'condition_numbers'])


# |A| != 0
def solve_using_regularization(A: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> SolutionUsingRegularization:
    if not np.linalg.det(A):
        raise ValueError('det(A) = 0')

    matrix = A + alpha * np.identity(len(b))

    return SolutionUsingRegularization(solve_using_LU_decomposition(matrix, b), calculate_condition_numbers(matrix))
