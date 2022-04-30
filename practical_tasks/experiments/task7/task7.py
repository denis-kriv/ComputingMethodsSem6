from math import cos, sin, exp, pi, log, e
from typing import Callable, Tuple, List, Dict

from common import create_plot
from practical_tasks.solutions.task7.task7 import solve_using_grid_method


def create_plots(experiments_results, splitting_power_range, names, a, b, equation):
    h_range = [(b - a) / splitting_power for splitting_power in splitting_power_range]
    delta_range = [experiments_results[0][h] for h in h_range]
    create_plot('results', f'{names[0]}.png', h_range, [(delta_range, '|max(y - y_grid)|')], equation, 'h', '||delta||')

    xs, y_r, y_grid, y = experiments_results[1]
    create_plot('results', f'{names[1]}.png', xs, [(y_r, 'y_richardson'), (y_grid, 'y_grid'), (y, 'y')], equation)


def experiment(splitting_power_range: List, y: Callable,
               p: Callable, q: Callable, r: Callable, f: Callable, a: float, b: float,
               alpha1: float, alpha2: float, alpha: float, beta1: float, beta2: float, beta: float,
               fst) -> Tuple[Dict, Tuple]:
    results, prev = {}, lambda _: 0
    for splitting_power in splitting_power_range:
        h = (b - a) / splitting_power
        y_grid = solve_using_grid_method(p, q, r, f, a, b, splitting_power, alpha1,
                                         alpha2, alpha, beta1, beta2, beta, fst)
        results[h] = max([abs(y_grid(x) - y(x)) for x in [a + h * i for i in range(splitting_power)]])

        if splitting_power == splitting_power_range[-1]:
            return results, tuple(zip(*[(x, y_grid(x) * 2 - prev(x), y_grid(x), y(x))
                                  for x in [a + i * h for i in range(splitting_power)]]))

        prev = y_grid


def experiment1(splitting_power_range: List):
    eq = "-u''/(x^3+5)+cos(x)*u'+e^x*u=sin(x)/(x^3+5)+cos(x)^2+e^x*sin(x), \nu(0)=0, u(pi)-u'(pi)=1, \n(u(x)=sin(x))"

    p = lambda x: 1 / (x ** 3 + 5)
    q = cos
    r = exp
    f = lambda x: sin(x) / (x ** 3 + 5) + cos(x) ** 2 + exp(x) * sin(x)
    a, b = 0, pi / 2
    alpha1, alpha2, alpha = 1, 0, 0
    beta1, beta2, beta = 1, 1, 1
    fst = True
    y = sin

    result = experiment(splitting_power_range, y, p, q, r, f, a, b, alpha1, alpha2, alpha, beta1, beta2, beta, fst)
    create_plots(result, splitting_power_range, ['equation1_delta_h', 'equation1'], a, b, eq)


def experiment2(splitting_power_range: List):
    eq = "-(10*u')'+x*u'+x*y=2*x^2+2*x, \nu(1)=2, u(2)=2, \n(u(x)=2*x)"
    p = lambda _: 10
    q = lambda x: x
    r = lambda x: x
    f = lambda x: 2 * x ** 2 + 2 * x
    a, b = 1, 2
    alpha1, alpha2, alpha = 1, 0, 2
    beta1, beta2, beta = 1, 0, 4
    fst = False
    y = lambda x: 2 * x

    result = experiment(splitting_power_range, y, p, q, r, f, a, b, alpha1, alpha2, alpha, beta1, beta2, beta, fst)
    create_plots(result, splitting_power_range, ['equation2_delta_h', 'equation2'], a, b, eq)


# -(p(x) * y')' + q(x) * y' + r(x) * y = f(x) , else
def experiment3(splitting_power_range: List):
    eq = "-(10*u')'+*u'+x*y=2*x^2+2*x, \nu(1)=2, u(2)=2, \n(u(x)=2*x)"
    p = lambda _: 10
    q = log
    r = lambda _: 11
    f = lambda x: 10 / x ** 2 + log(x) / x + 11 * log(x)
    a, b = 1, e
    alpha1, alpha2, alpha = 1, 0, 0
    beta1, beta2, beta = 2, 0, 2
    fst = False
    y = log

    result = experiment(splitting_power_range, y, p, q, r, f, a, b, alpha1, alpha2, alpha, beta1, beta2, beta, fst)
    create_plots(result, splitting_power_range, ['equation3_delta_h', 'equation3'], a, b, eq)


def run():
    splitting_power_range = [10 ** i for i in range(2, 5)]

    experiment1(splitting_power_range)
    experiment2(splitting_power_range)
    experiment3(splitting_power_range)


if __name__ == '__main__':
    run()
