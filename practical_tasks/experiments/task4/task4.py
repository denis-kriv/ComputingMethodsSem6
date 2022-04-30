from collections import namedtuple
from typing import Iterable, Callable, Dict

import numpy as np

from practical_tasks.experiments.common import generate_diagonal_dominance_matrix_element
from practical_tasks.solutions.task4.task4a import solve_using_simple_iterations_method
from practical_tasks.solutions.task4.task4b import solve_using_zeidels_method
from common import create_table, create_plot

Result = namedtuple('Result', ['epsilon', 'simple_iterations_method_iterations_count',
                               'zeidels_method_iterations_count'])


def experiment(dimension_range: Iterable, epsilon_range: Iterable, generate: Callable) -> Dict:
    results = {}

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])
        x = np.ones(dimension)
        b = A.dot(x)

        results[dimension] = [Result(epsilon, solve_using_simple_iterations_method(A, b, epsilon).iterations_count,
                                     solve_using_zeidels_method(A, b, epsilon).iterations_count)
                              for epsilon in epsilon_range]

    return results


def run():
    def unpack(experiments_results):
        return [[(n,) + row for row in experiments_results[n]] for n in experiments_results]

    def create_plots(experiments_results, file_name_template):
        for n in experiments_results:
            epsilons, simple_iterations, zeidels = zip(*experiments_results[n])
            create_plot('results', f'{file_name_template}={n}.png', epsilons,
                        [(simple_iterations, 'simple_iterations'), (zeidels, 'zeidels')],
                        f'n={n}', 'epsilon', 'iteration_count')

    dimension_range, epsilon_range = range(10, 20), [10 ** (-i) for i in range(6, 10)]
    titles = ('matrix_dimension',) + Result._fields

    diagonal_dominance_matrix_results = experiment(dimension_range, epsilon_range,
                                                   generate_diagonal_dominance_matrix_element)
    create_table('results', 'task4_diagonal_dominance_matrix_results.csv', titles,
                 unpack(diagonal_dominance_matrix_results))
    create_plots(diagonal_dominance_matrix_results, 'task4_diagonal_dominance_matrix_results_for_n')


if __name__ == '__main__':
    run()
