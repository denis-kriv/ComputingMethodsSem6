from itertools import product
from math import exp
from typing import Callable, Tuple, List

from common import create_plot
from practical_tasks.solutions.task8.task8 import solve_using_ritz_method


def d(f):
    h = 10 ** (-7)
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


def left(p, r, u):
    return lambda x: -d(lambda x1: p(x1) * d(u)(x1))(x) + (lambda x2: r(x2) * u(x2))(x)


def experiment(p: Callable, r: Callable, f: Callable, n: int,
               alpha1: float, alpha2: float, beta1: float, beta2: float, k: int,
               y: Callable, xs: List, equation: str, names: Tuple):
    y_ritz = solve_using_ritz_method(p, r, f, n, alpha1, alpha2, beta1, beta2, k)

    """create_plot('results', f'{names[0]}.png', xs,
                [([y_ritz(x) for x in xs], 'y_ritz'), ([y(x) for x in xs], 'y')], equation)"""
    create_plot('results', f'{names[1]}.png', xs,
                [([left(p, r, y_ritz)(x) for x in xs], 'y_ritz'),  # ([left(p, r, y)(x) for x in xs], 'y'),
                 ([f(x) for x in xs], 'f')], equation)


def experiment1(n: int, k: int):
    eq = "-(10*u')'+x^2*u=(x^2-10)*e^x, \nu(-1)-u'(-1)=u(1)-u'(1)=0"
    p = lambda _: 10
    r = lambda x: x ** 2
    f = lambda x: (x ** 2 - 10) * exp(x)
    alpha1, alpha2 = 1, 1
    beta1, beta2 = 1, 1
    y = exp

    experiment(p, r, f, n, alpha1, alpha2, beta1, beta2, k, y, [-1 + 2 * i / 99 for i in range(100)], eq,
               (f'experiment1_y_n={n}_k={k}', f'experiment1_full_n={n}_k={k}'))


def experiment2(n: int, k: int):
    eq = "-((x+2)*u')'+10*u=10*x^2-4*x-14, \nu(-1)=u(1)=0"
    p = lambda x: x + 2
    r = lambda _: 10
    f = lambda x: 10 * x ** 2 - 4 * x - 14
    alpha1, alpha2 = 1, 0
    beta1, beta2 = 1, 0
    y = lambda x: x ** 2 - 1

    experiment(p, r, f, n, alpha1, alpha2, beta1, beta2, k, y, [-1 + 2 * i / 99 for i in range(100)], eq,
               (f'experiment2_y_n={n}_k={k}', f'experiment2_full_n={n}_k={k}'))


def experiment3(n: int, k: int):
    eq = "-((x+2)*u')'+5*u=-10*x^2+4*x+9, \nu(-1)=u(1)=0"
    p = lambda x: x + 2
    r = lambda _: 5
    f = lambda x: -5 * x ** 2 + 4 * x + 9
    alpha1, alpha2 = 1, 0
    beta1, beta2 = 1, 0
    y = lambda x: -x ** 2 + 1

    experiment(p, r, f, n, alpha1, alpha2, beta1, beta2, k, y, [-1 + 2 * i / 99 for i in range(100)], eq,
               (f'experiment3_y_n={n}_k={k}', f'experiment3_full_n={n}_k={k}'))


def experiment4(n: int, k: int):
    eq = "-(x^3*u')'+(e^x)*u=e^x*(x-1)-3*x^2, \nu(-1)=u(1)-2*u'(-1)=0"
    p = lambda x: x ** 3 + 5
    r = exp
    f = lambda x: exp(x) * (x - 1) - 3 * x ** 2
    alpha1, alpha2 = 1, 0
    beta1, beta2 = 1, 2
    y = lambda x: x - 1

    experiment(p, r, f, n, alpha1, alpha2, beta1, beta2, k, y, [-1 + 2 * i / 99 for i in range(100)], eq,
               (f'experiment4_y_n={n}_k={k}', f'experiment4_full_n={n}_k={k}'))


def experiment5(n: int, k: int):
    eq = "-(u'/(x+2))'+x^2*u=0, \n-u'(-1)=-u'(1)=0"
    p = lambda x: 1 / (x + 2)
    r = lambda x: x ** 2
    f = lambda _: 0
    alpha1, alpha2 = 0, 1
    beta1, beta2 = 0, 1
    y = lambda _: 0

    experiment(p, r, f, n, alpha1, alpha2, beta1, beta2, k, y, [-1 + 2 * i / 99 for i in range(100)], eq,
               (f'experiment5_y_n={n}_k={k}', f'experiment5_full_n={n}_k={k}'))


def run():
    for n, k in product([5, 10, 20], [0, 3, 7]):
        experiment1(n, k)
        experiment2(n, k)
        experiment3(n, k)
        experiment4(n, k)
        experiment5(n, k)


if __name__ == '__main__':
    run()
