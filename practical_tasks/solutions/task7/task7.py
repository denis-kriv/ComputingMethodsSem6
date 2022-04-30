from functools import reduce
from typing import Callable

import numpy as np


# p, q, f -- smooth, p(x) >= p0 > 0, r(x) >= 0, where a <= x <= b (WITHOUT CHECKING)
#
# -p(x)  * y''  + q(x) * y' + r(x) * y = f(x) , if fst
# -(p(x) * y')' + q(x) * y' + r(x) * y = f(x) , else
# a < x < b
#
# alpha1 * y(a) - alpha2 * y'(a) = alpha, |alpha1| + |alpha2| != 0, alpha1 * alpha2 >= 0
# beta1  * y(b) - beta2  * y'(b) = beta , |beta1|  + |beta2|  != 0, beta1  * beta2  >= 0
def solve_using_grid_method(p: Callable, q: Callable, r: Callable, f: Callable, a: float, b: float, n: int,
                            alpha1: float, alpha2: float, alpha: float, beta1: float, beta2: float, beta: float,
                            fst=True) -> Callable:
    h = (b - a) / n
    X = [a + h * i / 2 for i in range(2 * n + 1)]
    p_i, q_i, r_i, f_i = zip(*[(p(x), q(x), r(x), f(x)) for x in X])

    if fst:
        A, B, C, G = zip(*[(-(p_i[i] / h + q_i[i] / 2) / h, 2 * p_i[i] / h ** 2 + r_i[i],
                            (-p_i[i] / h + q_i[i] / 2) / h, f_i[i]) for i in range(2, 2 * n, 2)])
    else:
        A, B, C, G = zip(*[(-(p_i[i - 1] / h + q_i[i] / 2) / h, (p_i[i + 1] + p_i[i - 1]) / h ** 2 + r_i[i],
                            (-p_i[i + 1] / h + q_i[i] / 2) / h, f_i[i]) for i in range(2, 2 * n, 2)])
    A, B, C, G = [0.0] + list(A) + [-beta2 / h], [alpha1 + alpha2 / h] + list(B) + [beta1 + beta2 / h], \
                 [-alpha2 / h] + list(C) + [0.0], [alpha] + list(G) + [beta]

    M = np.array([[A[row] if row == column + 1 else B[row] if row == column else C[row] if row == column - 1 else 0
                   for column in range(n + 1)] for row in range(n + 1)])
    v = np.array(G)
    Y = np.linalg.solve(M, v)

    return lambda x: reduce(lambda y, i: Y[i // 2 + int(x - X[i] < X[i + 2] - x)] if X[i] < x <= X[i + 2] else y,
                            range(0, 2 * n, 2), Y[0])
