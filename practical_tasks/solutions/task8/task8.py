from typing import Callable

import numpy as np
from scipy import integrate
from scipy.special import eval_jacobi


def scalar_multiplication(f1, f2):
    return integrate.quad(lambda x: f1(x) * f2(x), -1, 1)[0]


def bilinear_form(f1, f2, df1, df2, p, r, Q):
    return integrate.quad(lambda x: p(x) * df1(x) * df2(x) + r(x) * f1(x) * f2(x), -1, 1)[0] + Q(f1, f2)


def get_Jacobi_polynomials_and_derivatives(n, k):
    P = [(lambda N: (lambda x: eval_jacobi(N, k + 1, k + 1, x)))(i) for i in range(n - 1)]
    return [(lambda N: (lambda x: eval_jacobi(N, k, k, x)))(i) for i in range(n)], \
           [(lambda N: (lambda _: 0) if N == 0 else (lambda x: (N + 2 * k + 1) * P[N - 1](x) / 2))(i) for i in range(n)]


# p -- smooth, r(x) -- continuous, p(x) >= p0 > 0, r(x) > 0 (WITHOUT CHECKING)
#
# -(p(x) * u')' + r(x) * u = f(x)
# a < x < b
#
# alpha1 * u(-1) - alpha2 * u'(-1) = 0, |alpha1| + |alpha2| != 0, alpha1 * alpha2 >= 0
# beta1  * u(1)  - beta2  * u'(1)  = 0, |beta1|  + |beta2|  != 0, beta1  * beta2  >= 0
#
# k -- index of Jacobi polynomials
def solve_using_ritz_method(p: Callable, r: Callable, f: Callable, n: int,
                            alpha1: float, alpha2: float, beta1: float, beta2: float, k=0) -> Callable:
    W, dW = get_Jacobi_polynomials_and_derivatives(n, k)

    Q = (lambda _, __: 0) if alpha1 == 0 or alpha2 == 0 else \
        (lambda f1, f2: alpha1 * p(-1) * f1(-1) * f2(-1) / alpha2 + beta1 * p(1) * f1(1) * f2(1) / beta2)
    M = np.array([[bilinear_form(W[row], W[column], dW[row], dW[column], p, r, Q)
                   for column in range(n)] for row in range(n)])
    v = np.array([scalar_multiplication(f, w) for w in W])
    C = np.linalg.solve(M, v)

    return lambda x: sum([C[i] * W[i](x) for i in range(n)])
