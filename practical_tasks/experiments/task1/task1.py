from collections import namedtuple
from typing import Iterable, Callable, Dict

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task1.task1 import calculate_condition_numbers
from common import create_table, create_plot

Result = namedtuple('Result', ['decimal_places', 'error_norm', 'condition_numbers'])


def experiment(dimension_range: Iterable, decimal_places_range: Iterable, generate: Callable) -> Dict:
    results = {}

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])
        x = np.ones(dimension)
        b = A.dot(x)

        results[dimension] = []
        for decimal_places in decimal_places_range:
            A_rounded = np.array([[round(element, decimal_places) for element in row] for row in A])
            b_rounded = np.array([round(element, decimal_places) for element in b])
            results[dimension].append(Result(decimal_places, np.linalg.norm(x - np.linalg.solve(A_rounded, b_rounded)),
                                             calculate_condition_numbers(A_rounded)))

    return results


def run():
    def unpack(experiment_results):
        unpacking = []
        for n in experiment_results:
            unpacking += [(n, result.decimal_places, result.error_norm, result.condition_numbers.spectral,
                           result.condition_numbers.volume, result.condition_numbers.angular)
                          for result in experiment_results[n]]
        return unpacking

    def create_plots(experiment_results, spectral_file_name_template,
                     volume_file_name_template, angular_file_name_template):
        for n in experiment_results:
            error_norms, spectral, volume, angular = zip(*[(row.error_norm,) + row.condition_numbers
                                                           for row in experiment_results[n]])
            create_plot('results', f'{spectral_file_name_template}={n}.png', spectral, [(error_norms, '||x_0 - x||')],
                        f'dependence between spectral condition number and ||x_0 - x||, n={n}', 'condition_number')
            create_plot('results', f'{volume_file_name_template}={n}.png', volume, [(error_norms, '||x_0 - x||')],
                        f'dependence between volume condition number and ||x_0 - x||, n={n}', 'condition_number')
            create_plot('results', f'{angular_file_name_template}={n}.png', angular, [(error_norms, '||x_0 - x||')],
                        f'dependence between angular condition number and ||x_0 - x||, n={n}', 'condition_number')

    dimension_range = range(2, 8)
    decimal_places_range = range(2, 8)
    titles = ('matrix_dimension', 'decimal_places', '||x_0 - x||', 'spectral_condition_number',
              'volume_condition_number', 'angular_condition_number')

    hilberts_results = experiment(dimension_range, decimal_places_range, generate_hilberts_matrix_element)
    create_table('results', 'task1_hilberts_matrix_results.csv', titles, unpack(hilberts_results))
    create_plots(hilberts_results, 'task1_hilberts_matrix_results_plot_for_spectral_n',
                 'task1_hilberts_matrix_results_plot_for_volume_n', 'task1_hilberts_matrix_results_plot_for_angular_n')

    tridiagonal_results = experiment(dimension_range, decimal_places_range, generate_tridiagonal_matrix_element)
    create_table('results', 'task1_tridiagonal_matrix_results.csv', titles, unpack(tridiagonal_results))
    # create_plots(tridiagonal_results, 'task1_tridiagonal_matrix_results_plot_for_spectral_n',
    #             'task1_tridiagonal_matrix_results_plot_for_volume_n',
    #             'task1_tridiagonal_matrix_results_plot_for_angular_n')


if __name__ == '__main__':
    run()
