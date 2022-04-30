import unittest
from math import exp

from practical_tasks.solutions.task8.task8 import solve_using_ritz_method


class TestTask8(unittest.TestCase):
    def test_solve_using_ritz_method_should_works_correctly_with_example_1(self):
        n = 3
        p = lambda x: x + 2
        r = lambda _: 10
        f = lambda x: -exp(x) * (x + 3) + 10 * exp(x)
        alpha1, alpha2 = 1, 1
        beta1, beta2 = 1, 1

        actual = solve_using_ritz_method(p, r, f, n, alpha1, alpha2, beta1, beta2)
        expected = exp

        self.assertTrue(all([abs(actual(x) - expected(x)) < 3 for x in [-1 + 2 * i / 10 for i in range(n)]]))

    def test_solve_using_ritz_method_should_works_correctly_with_example_2(self):
        n = 3
        p = lambda x: 1 / (x + 2)
        r = lambda x: x ** 2
        f = lambda _: 0
        alpha1, alpha2 = 1, 0
        beta1, beta2 = 1, 0

        actual = solve_using_ritz_method(p, r, f, n, alpha1, alpha2, beta1, beta2)
        expected = lambda _: 0

        self.assertTrue(all([abs(actual(x) - expected(x)) < 1 for x in [-1 + 2 * i / 10 for i in range(n)]]))
