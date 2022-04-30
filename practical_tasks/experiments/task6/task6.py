from collections import namedtuple
from functools import reduce
from typing import Iterable, Callable, Dict

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element
from practical_tasks.solutions.task6.task6a import find_eigen_values_using_first_method
from practical_tasks.solutions.task6.task6b import find_eigen_values_using_second_method

from common import create_table, create_plot

Result = namedtuple('Result', ['epsilon', 'first_method_iterations_count', 'first_method_errors_count',
                               'second_method_iterations_count', 'second_method_errors_count'])


def experiment(dimension_range: Iterable, epsilon_range: Iterable, max_iterations_count: int,
               generate: Callable) -> Dict:
    results = {}

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])

        results[dimension] = []
        for epsilon in epsilon_range:
            first_result = find_eigen_values_using_first_method(A, epsilon, max_iterations_count)
            second_result = find_eigen_values_using_second_method(A, epsilon, max_iterations_count)

            Rs = [sum([abs(A[row, column]) for column in range(dimension)]) - abs(A[row, row])
                  for row in range(dimension)]
            first_errors, second_errors = 0, 0
            for fst, snd in zip(first_result.eigen_values, second_result.eigen_values):
                f1, f2 = reduce(lambda r, i: (r[0] or abs(fst - A[i, i]) <= Rs[i], r[1] or abs(snd - A[i, i]) <= Rs[i]),
                                range(dimension), (False, False))
                first_errors, second_errors = first_errors + int(not f1), second_errors + int(not f2)

            results[dimension].append(Result(epsilon, first_result.iterations_count, first_errors,
                                             second_result.iterations_count, second_errors))

    return results


def run():
    def unpack(experiments_results):
        return [[(n,) + row for row in experiments_results[n]] for n in experiments_results]

    def create_plots(experiments_results, file_name_template):
        for n in experiments_results:
            epsilons, first, _, second, _ = zip(*experiments_results[n])
            create_plot('results', f'{file_name_template}={n}.png', epsilons,
                        [(first, 'first'), (second, 'second')],
                        f'n={n}', 'epsilon', 'iteration_count')

    dimension_range, epsilon_range, max_iterations_count = range(10, 20), [10 ** (-i) for i in range(6, 10)], 100000
    titles = ('matrix_dimension',) + Result._fields

    hilberts_matrix_results = experiment(dimension_range, epsilon_range, max_iterations_count,
                                         generate_hilberts_matrix_element)
    create_table('results', 'task6_hilberts_matrix_results.csv', titles, unpack(hilberts_matrix_results))
    create_plots(hilberts_matrix_results, 'task6_hilberts_matrix_results_for_n')


if __name__ == '__main__':
    run()
