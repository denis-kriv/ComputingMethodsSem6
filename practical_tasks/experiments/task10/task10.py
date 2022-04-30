from itertools import product

from common import create_plot
from practical_tasks.solutions.task10.task10a import solve_using_explicit_method
from practical_tasks.solutions.task10.task10b import solve_using_implicit_method


def experiment1(f, kappa, a, T, N_range, K, mu, mu1, mu2, u, equation):
    tau = T / (K - 1)
    ts = [k * tau for k in range(K)]
    delta_explicit, delta_implicit = [], []
    for N in N_range:
        u_explicit = solve_using_explicit_method(f, kappa, a, T, N, K, mu, mu1, mu2)
        u_implicit = solve_using_implicit_method(f, kappa, a, T, N, K, mu, mu1, mu2)

        h = a / (N - 1)
        xs = [n * h for n in range(N)]
        delta_explicit.append(max([abs(u_explicit[j][i] - u(xs[i], ts[j]))
                                   for i, j in product(range(N), range(K))]))
        delta_implicit.append(max([abs(u_implicit[j][i] - u(xs[i], ts[j]))
                                   for i, j in product(range(N), range(K))]))

    create_plot('results', f'task10_delta_N_explicit.png', N_range, [(delta_explicit, '|max(u - u_explicit)|')],
                equation, 'N', '||delta||')
    create_plot('results', f'task10_delta_N_implicit.png', N_range, [(delta_implicit, '|max(u - u_implicit)|')],
                equation, 'N', '||delta||')


def experiment2(f, kappa, a, T, N, K_range, mu, mu1, mu2, u, equation):
    h = a / (N - 1)
    xs = [n * h for n in range(N)]
    delta_explicit, delta_implicit = [], []
    for K in K_range:
        u_explicit = solve_using_explicit_method(f, kappa, a, T, N, K, mu, mu1, mu2)
        u_implicit = solve_using_implicit_method(f, kappa, a, T, N, K, mu, mu1, mu2)

        tau = T / (K - 1)
        ts = [k * tau for k in range(K)]

        delta_explicit.append(max([abs(u_explicit[j][i] - u(xs[i], ts[j]))
                                   for i, j in product(range(N), range(K))]))
        delta_implicit.append(max([abs(u_implicit[j][i] - u(xs[i], ts[j]))
                                   for i, j in product(range(N), range(K))]))

    create_plot('results', f'task10_delta_K_explicit.png', K_range, [(delta_explicit, '|max(u - u_explicit)|')],
                equation, 'K', '||delta||')
    create_plot('results', f'task10_delta_K_implicit.png', K_range, [(delta_implicit, '|max(u - u_implicit)|')],
                equation, 'K', '||delta||')


def run():
    eq = "du/dt=2*(du/d^2x)-12*x+1, \nu(x,0)=x^3,u(0,t)=t,u(1,t)=1+t, where 0 <= x <= a, 0 <= t <= T \n(u(x,t)=x^3+t)"

    ranges, default = [5, 10, 15], 10

    f = lambda x, _: -12 * x + 1
    kappa = 2
    a, T = 1, 1
    mu = lambda x: x ** 3
    mu1 = lambda t: t
    mu2 = lambda t: a ** 3 + t
    u = lambda x, t: x ** 3 + t

    experiment1(f, kappa, a, T, ranges, default, mu, mu1, mu2, u, eq)
    experiment2(f, kappa, a, T, default, ranges, mu, mu1, mu2, u, eq)


if __name__ == '__main__':
    run()
