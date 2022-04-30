from typing import Callable, List


# kappa > 0
#
# du(x, t) / dt = kappa * (du(x, t) / d^2x) + f(x, t)
# 0 < x < a, 0 < t <= T
#
# u(x, 0) = mu(x),                    where 0 <= x <= a
# u(0, t) = mu1(t), u(a, t) = mu2(t), where 0 <= t <= T
def solve_using_explicit_method(f: Callable, kappa: float, a: float, T: float, N: int, K: int,
                                mu: Callable, mu1: Callable, mu2: Callable) -> List:
    if kappa <= 0:
        raise ValueError('kappa <= 0.')

    h, tau = a / (N - 1), T / (K - 1)
    xs, ts = [n * h for n in range(N)], [k * tau for k in range(K)]
    koef = kappa * tau / h ** 2

    u_levels = [[mu(x) for x in xs]]
    for i in range(1, K):
        u = u_levels[-1]
        u_levels.append([mu1(ts[i]) if n == 0 else
                         mu2(ts[i]) if n == N - 1 else
                         koef * (u[n - 1] - 2 * u[n] + u[n + 1]) + u[n] + tau * f(xs[n], ts[i - 1]) for n in range(N)])

    return u_levels
