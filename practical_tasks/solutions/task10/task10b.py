from typing import Callable, List

import numpy as np


# kappa > 0
#
# du(x, t) / dt = kappa * (du(x, t) / d^2x) + f(x, t)
# 0 < x < a, 0 < t <= T
#
# u(x, 0) = mu(x),                    where 0 <= x <= a
# u(0, t) = mu1(t), u(a, t) = mu2(t), where 0 <= t <= T
def solve_using_implicit_method(f: Callable, kappa: float, a: float, T: float, N: int, K: int,
                                mu: Callable, mu1: Callable, mu2: Callable) -> List:
    if kappa <= 0:
        raise ValueError('kappa <= 0.')

    h, tau = a / (N - 1), T / (K - 1)
    xs, ts = [n * h for n in range(N)], [k * tau for k in range(K)]

    k1, k2 = -kappa * tau / h ** 2, 2 * kappa * tau / h ** 2 + 1
    M = np.array([[(1 if row == column else 0) if row == 0 or row == N - 1 else
                   k1 if abs(row - column) == 1 else
                   k2 if row == column else 0 for column in range(N)] for row in range(N)])

    u_levels = [[mu(x) for x in xs]]
    for t in ts[1:]:
        u_levels.append(np.linalg.solve(M, np.array([mu1(t) if n == 0 else mu2(t) if n == N - 1 else
                                                     u_levels[-1][n] + tau * f(xs[n], t)
                                                     for n in range(N)])))

    return u_levels
