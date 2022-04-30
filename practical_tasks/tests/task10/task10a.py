import unittest
from itertools import product

from practical_tasks.solutions.task10.task10a import solve_using_explicit_method


class TestTask10a(unittest.TestCase):
    def test_solve_using_explicit_method_should_works_correctly_with_example_1(self):
        f = lambda x, _: -0.006 * x + 1
        kappa = 0.001
        a, T, N, K = 1, 1, 100, 5
        mu = lambda x: x ** 3
        mu1 = lambda t: t
        mu2 = lambda t: a ** 3 + t

        actual = solve_using_explicit_method(f, kappa, a, T, N, K, mu, mu1, mu2)
        expected = lambda x, t: x ** 3 + t

        h, tao = a / (N - 1), T / (K - 1)
        xs, ts = [n * h for n in range(N)], [k * tao for k in range(K)]

        self.assertTrue(all([abs(actual[j][i] - expected(xs[i], ts[j])) < 1
                             for i, j in product(range(N), range(K))]))
