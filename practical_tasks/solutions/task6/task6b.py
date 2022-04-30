from typing import Tuple

import numpy as np

from practical_tasks.solutions.task6.common import EigenValues, find_eigen_values


# a_{ij} -- next in order
def get_element(A_k: np.ndarray, i: int, j: int) -> Tuple[int, int]:
    n = A_k.shape[0]

    return (i, j + 1) if j < n - 1 and i != j + 1 else (i, j + 2) if j < n - 2 else (i + 1, 0) if i < n - 1 else (0, 1)


# A = A*
def find_eigen_values_using_second_method(A: np.ndarray, epsilon: float, max_iterations_count: int) -> EigenValues:
    return find_eigen_values(A, epsilon, max_iterations_count, get_element)
