import unittest
from math import cos, sin, exp, pi, log, e

from practical_tasks.solutions.task7.task7 import solve_using_grid_method


class TestTask7(unittest.TestCase):
    def test_solve_using_grid_method_should_works_correctly_with_example_1(self):
        n = 1000

        p = lambda x: 1 / (x ** 3 + 5)
        q = cos
        r = exp
        f = lambda x: sin(x) / (x ** 3 + 5) + cos(x) ** 2 + exp(x) * sin(x)
        a, b = 0, pi / 2
        alpha1, alpha2, alpha = 1, 0, 0
        beta1, beta2, beta = 1, 1, 1

        h = (b - a) / n

        actual = solve_using_grid_method(p, q, r, f, a, b, n, alpha1, alpha2, alpha, beta1, beta2, beta)
        expected = sin

        self.assertTrue(all([abs(actual(x) - expected(x)) < h for x in [a + h * i for i in range(n)]]))

    def test_solve_using_grid_method_should_works_correctly_with_example_2(self):
        n = 1000

        p = lambda _: 10
        q = lambda x: x
        r = lambda x: x
        f = lambda x: 2 * x ** 2 + 2 * x
        a, b = 1, 2
        alpha1, alpha2, alpha = 1, 0, 2
        beta1, beta2, beta = 1, 0, 4

        h = (b - a) / n

        actual = solve_using_grid_method(p, q, r, f, a, b, n, alpha1, alpha2, alpha, beta1, beta2, beta, False)
        expected = lambda x: 2 * x

        self.assertTrue(all([abs(actual(x) - expected(x)) < 3 * h for x in [a + h * i for i in range(n)]]))

    def test_solve_using_grid_method_should_works_correctly_with_example_3(self):
        n = 1000

        p = lambda _: 10
        q = log
        r = lambda _: 11
        f = lambda x: 10 / x ** 2 + log(x) / x + 11 * log(x)
        a, b = 1, e
        alpha1, alpha2, alpha = 1, 0, 0
        beta1, beta2, beta = 2, 0, 2

        h = (b - a) / n

        actual = solve_using_grid_method(p, q, r, f, a, b, n, alpha1, alpha2, alpha, beta1, beta2, beta, False)
        expected = log

        self.assertTrue(all([abs(actual(x) - expected(x)) < h for x in [a + h * i for i in range(n)]]))
