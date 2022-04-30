from collections import namedtuple
from typing import Iterable, Callable, Dict

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element
from practical_tasks.solutions.task5.task5a import find_using_power_method
from practical_tasks.solutions.task5.task5b import find_using_scalar_product_method

from common import create_table, create_plot

Result = namedtuple('Result', ['epsilon', 'power_method_iterations_count', 'scalar_product_method_iterations_count'])


def experiment(dimension_range: Iterable, epsilon_range: Iterable, max_iterations_count: int,
               generate: Callable) -> Dict:
    results = {}

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])

        results[dimension] = \
            [Result(epsilon, find_using_power_method(A, epsilon, max_iterations_count).iterations_count,
                    find_using_scalar_product_method(A, epsilon, max_iterations_count).iterations_count)
             for epsilon in epsilon_range]

    return results


def run():
    def unpack(experiments_results):
        return [[(n,) + row for row in experiments_results[n]] for n in experiments_results]

    def create_plots(experiments_results, file_name_template):
        for n in experiments_results:
            epsilons, power, scalar_product = zip(*experiments_results[n])
            create_plot('results', f'{file_name_template}={n}.png', epsilons,
                        [(power, 'power'), (scalar_product, 'scalar_product')],
                        f'n={n}', 'epsilon', 'iteration_count')

    dimension_range, epsilon_range, max_iterations_count = range(10, 20), [10 ** (-i) for i in range(6, 10)], 100000
    titles = ('matrix_dimension',) + Result._fields

    hilberts_matrix_results = experiment(dimension_range, epsilon_range, max_iterations_count,
                                         generate_hilberts_matrix_element)
    create_table('results', 'task5_hilberts_matrix_results.csv', titles, unpack(hilberts_matrix_results))
    create_plots(hilberts_matrix_results, 'task5_hilberts_matrix_results_for_n')


if __name__ == '__main__':
    run()
