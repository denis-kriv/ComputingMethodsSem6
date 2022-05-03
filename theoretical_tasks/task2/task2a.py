from math import cos, sin
from typing import Callable, List, Tuple, Iterable

import numpy as np

from common import create_plot


# Ax=b -> (conjugate(A)*A+alpha*E)=conjugate(A)*b
def solve_using_regularization(A: np.ndarray, b: np.ndarray, alphas: List[float]) -> List[np.ndarray]:
    n = A.shape[0]
    A_conjugate = np.transpose(np.conjugate(A))
    return [np.linalg.solve(A_conjugate.dot(A) + alpha * np.identity(n), A_conjugate.dot(b)) for alpha in alphas]


# I_{0}^{1}(K(x,s)*z(s))ds=u(x), 0<=x<=1
def solve_using_first_method(K: Callable[[float, float], float], u: Callable[[float], float],
                             n: int, alphas: List[float]) -> Tuple[List[float], List[np.ndarray]]:
    S = [(k + 0.5) / n for k in range(n)]
    C, U = np.array([[K(x, s) / n for s in S] for x in S]), np.array([u(s) for s in S])

    return S, solve_using_regularization(C, U, alphas)


def experiment(K: Callable[[float, float], float], u: Callable[[float], float], z: Callable[[float], float],
               n_range: Iterable[int], alphas: List[float], equation: str, file_name_template: str):
    for n in n_range:
        S, Z_reg = solve_using_first_method(K, u, n, alphas)
        Z = np.array([z(s) for s in S])
        create_plot('results', f'{file_name_template} for n={n}.png', alphas,
                    [([np.linalg.norm(Z - Z_reg[i]) for i in range(len(alphas))], '||z - z_reg||')],
                    equation, 'alpha', 'error_norm', 'upper right')


def run():
    n_range = [100, 500, 1000]
    big_alphas_range = [10 ** (-i) for i in range(1, 16)]
    small_alphas_range = [i * 50 * 10 ** (-15) for i in range(1, 20)]
    K = lambda x, s: cos(1 - s * x)

    z1, u1 = lambda _: 1, lambda x: (sin(1) - sin(1 - x)) / x
    equation1 = 'K(x,s)=cos(1-s*x), u(x)=(sin(1)-sin(1-x))/x \n(z(s)=1)'
    experiment(K, u1, z1, n_range, big_alphas_range, equation1, 'task2a_first_experiment_with_big_range')
    experiment(K, u1, z1, n_range, small_alphas_range, equation1, 'task2a_first_experiment_with_small_range')

    z2, u2 = lambda s: s * (1 - s), lambda x: -2 * cos(1 - x / 2) * (x * cos(x / 2) - 2 * sin(x / 2)) / x ** 3
    equation2 = 'K(x,s)=cos(1-s*x), u(x)=-2*cos(1-x/2)*(x*cos(x/2)-2*sin(x/2))/x^3 \n(z(s)=s*(1-s))'
    experiment(K, u2, z2, n_range, big_alphas_range, equation2, 'task2a_second_experiment_with_big_range')
    experiment(K, u2, z2, n_range, small_alphas_range, equation2, 'task2a_second_experiment_with_small_range')


if __name__ == '__main__':
    run()
