from functools import reduce
from itertools import product
from typing import Tuple

import numpy as np

from practical_tasks.solutions.task6.common import EigenValues, find_eigen_values


# a_{ij} -- max by absolute value, where i != j
def get_element(A_k: np.ndarray, _: int, __: int) -> Tuple[int, int]:
    return reduce(lambda r, v: v if v[0] != v[1] and abs(A_k[r]) < abs(A_k[v]) else r,
                  product(range(A_k.shape[0]), range(A_k.shape[0])), (0, 1))


# A = A*
def find_eigen_values_using_first_method(A: np.ndarray, epsilon: float, max_iterations_count: int) -> EigenValues:
    return find_eigen_values(A, epsilon, max_iterations_count, get_element)
