from collections import namedtuple
from functools import reduce
from math import sqrt, inf

import numpy as np

ConditionNumbers = namedtuple('ConditionNumbers', ['spectral', 'volume', 'angular'])


# det(A) != 0
def calculate_condition_numbers(A: np.ndarray) -> ConditionNumbers:
    if not np.linalg.det(A):
        return ConditionNumbers(inf, inf, inf)

    A_inverse = np.linalg.inv(A)

    spectral = np.linalg.norm(A) * np.linalg.norm(A_inverse)
    volume = reduce(lambda m, row: m * sqrt(reduce(lambda s, element: s + element ** 2, row, 0)),
                    A, 1) / abs(np.linalg.det(A))
    angular = max([np.linalg.norm(A[n]) * np.linalg.norm(A_inverse[:, n]) for n in range(A.shape[0])])

    return ConditionNumbers(spectral, volume, angular)
