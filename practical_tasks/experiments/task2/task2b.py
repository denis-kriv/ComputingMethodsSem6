from collections import namedtuple
from functools import reduce
from typing import Iterable, Callable, Dict

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task2.task2b import solve_using_regularization
from common import create_table, create_plot

Result = namedtuple('Result', ['alpha_to_error_norm_and_condition_numbers', 'best_alpha', 'best_alpha_check'])


def update_info(A, x, b, alpha, best_alpha_info, alpha_to_error_norm_and_condition_numbers=None):
    solution = solve_using_regularization(A, b, alpha)
    error_norm = np.linalg.norm(solution.x - x)

    if alpha_to_error_norm_and_condition_numbers is not None:
        alpha_to_error_norm_and_condition_numbers[alpha] = (error_norm, solution.condition_numbers)

    difference = error_norm  # np.linalg.norm(A.dot(solution.x) - b) + alpha * error_norm

    return (alpha, difference) if not best_alpha_info[0] or best_alpha_info[1] > difference else best_alpha_info


def experiment(dimension_range: Iterable, alpha_range: Iterable, generate: Callable) -> Dict:
    results = {}

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])
        x, y = np.random.rand(dimension), np.random.rand(dimension)
        b, c = A.dot(x), A.dot(y)

        x_best_alpha, y_best_alpha, alpha_to_error_norm_and_condition_numbers = \
            reduce(lambda t, alpha: (update_info(A, x, b, alpha, t[0], t[2]), update_info(A, y, c, alpha, t[1]), t[2]),
                   alpha_range, ((None, None), (None, None), {}))

        results[dimension] = Result(alpha_to_error_norm_and_condition_numbers,
                                    x_best_alpha[0], x_best_alpha[0] == y_best_alpha[0])

    return results


def run():
    def unpack(experiment_results):
        result = []

        for n in experiment_results:
            d = experiment_results[n].alpha_to_error_norm_and_condition_numbers
            for alpha in d:
                error_norm, spectral, volume, angular = (d[alpha][0],) + d[alpha][1]
                result.append((n, alpha, error_norm, spectral, volume, angular, experiment_results[n].best_alpha,
                               experiment_results[n].best_alpha_check))

        return result

    def create_plots(experiment_results, error_file_name_template, numbers_file_name_template=None):
        for n in experiment_results:
            d = experiment_results[n].alpha_to_error_norm_and_condition_numbers
            error_norms, spectral, volume, angular = zip(*[(d[alpha][0],) + d[alpha][1] for alpha in alpha_range])
            create_plot('results', f'{error_file_name_template}={n}.png', alpha_range,
                        [(error_norms, '||x - x_reg||')], f'n={n}', 'alpha')
            if numbers_file_name_template:
                create_plot('results', f'{numbers_file_name_template}={n}.png', alpha_range,
                            [(spectral, 'spectral'), (volume, 'volume'), (angular, 'angular')], f'n={n}', 'alpha')

    dimension_range = range(20, 21)
    alpha_range = [10 ** (-i) for i in range(2, 10)]
    titles = ['matrix_dimension', 'alpha', '||x - x_reg||', 'spectral_condition_number', 'volume_condition_number',
              'angular_condition_number', 'best_alpha', 'best_alpha_check']

    hilberts_results = experiment(dimension_range, alpha_range, generate_hilberts_matrix_element)
    create_table('results', 'task2b_hilberts_matrix_results.csv', titles, unpack(hilberts_results))
    create_plots(hilberts_results, 'task2b_hilberts_matrix_results_for_n',
                 'task2b_hilberts_matrix_results_condition_numbers_for_n')

    tridiagonal_results = experiment(dimension_range, alpha_range, generate_tridiagonal_matrix_element)
    create_table('results', 'task2b_tridiagonal_matrix_results.csv', titles, unpack(tridiagonal_results))
    create_plots(tridiagonal_results, 'task2b_tridiagonal_matrix_results_for_n',
                 'task2b_tridiagonal_matrix_results_condition_numbers_for_n')


if __name__ == '__main__':
    run()
