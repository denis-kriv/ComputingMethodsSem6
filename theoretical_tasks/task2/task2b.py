from itertools import product
from math import cos, sin
from typing import Callable, List, Tuple, Iterable

import numpy as np
from scipy import integrate

from common import create_plot

from theoretical_tasks.task2.task2a import solve_using_regularization


# I_{0}^{1}(K(s, t)*z(t))dt=u(s), 0<=s<=1
# z(0)=z(1)=0
def solve_using_second_method(K: Callable[[float, float], float], u: Callable[[float], float], n: int, alpha: float,
                              p: Callable[[float], float], r: Callable[[float], float],
                              regularization_alphas: List[float]) -> Tuple[List[float], List[np.ndarray]]:
    K1 = lambda s, t: integrate.quad(lambda x: K(x, s) * K(x, t), 0, 1)[0]
    S = [(k + 0.5) / n + 1 for k in range(n + 1)]
    T, h = np.linspace(0, 1, n + 1, retstep=True)

    C, U = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
    for k in range(n + 1):
        if k == 0 or k == n:
            for j in range(n + 1):
                C[k, j] = 1 if k == j else 0
            U[k] = 0
        else:
            for j in range(n + 1):
                C[k, j] = (-alpha * p(T[k] - h / 2) / h ** 2 if k - 1 == j else
                           alpha * ((p(T[k] - h / 2) + p(T[k] + h / 2)) / h ** 2 + r(T[k])) if k == j else
                           -alpha * p(T[k] + h / 2) / h ** 2 if k + 1 == j else 0) + K1(S[k], T[j]) / n
            U[k] = integrate.quad(lambda t: K(t, S[k]) * u(t), 0, 1)[0]

    return T, solve_using_regularization(C, U, regularization_alphas)


def experiment(K: Callable[[float, float], float], u: Callable[[float], float], z: Callable[[float], float],
               n_range: Iterable[int], alphas: List[float], ps: List[Tuple[Callable[[float], float], str]],
               rs: List[Tuple[Callable[[float], float], str]], regularization_alphas: List[float],
               equation: str, file_name_template: str):
    for n, alpha, p, r in product(n_range, alphas, ps, rs):
        T, Z_reg = solve_using_second_method(K, u, n, alpha, p[0], r[0], regularization_alphas)
        Z = np.array([z(t) for t in T])
        create_plot('results', f'{file_name_template} for n={n},alpha={alpha},p(t)={p[1]},r(t)={r[1]}.png',
                    regularization_alphas,
                    [([np.linalg.norm(Z - Z_reg[i]) for i in range(len(regularization_alphas))], '||z - z_reg||')],
                    equation, 'alpha', 'error_norm', 'upper right')


def run():
    n_range = [500, 1000]
    alphas = [10 ** -15, 10 ** -8]
    ps = [(lambda _: 1, '1'), (lambda _: 1000, '1000')]
    rs = [(lambda _: 1, '1'), (lambda _: 2, '2000')]
    regularization_alphas_big_range = [10 ** (-i) for i in range(1, 16)]
    regularization_alphas_small_range = [i * 50 * 10 ** (-15) for i in range(1, 20)]
    K = lambda s, t: cos(1 - t * s)

    z, u = lambda t: t * (1 - t), lambda s: -2 * cos(1 - s / 2) * (s * cos(s / 2) - 2 * sin(s / 2)) / s ** 3
    equation = 'K(x,s)=cos(1-s*x), u(x)=-2*cos(1-x/2)*(x*cos(x/2)-2*sin(x/2))/x^3 \n(z(s)=s*(1-s))'
    experiment(K, u, z, n_range, alphas, ps, rs, regularization_alphas_big_range, equation,
               'task2b_experiment_with_big_range')
    experiment(K, u, z, n_range, alphas, ps, rs, regularization_alphas_small_range, equation,
               'task2b_experiment_with_small_range')


if __name__ == '__main__':
    run()
