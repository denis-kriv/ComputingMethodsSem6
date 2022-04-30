from collections import namedtuple
from typing import Iterable, Callable, List

import numpy as np

from practical_tasks.experiments.common import generate_hilberts_matrix_element, generate_tridiagonal_matrix_element
from practical_tasks.solutions.task1.task1 import calculate_condition_numbers
from practical_tasks.solutions.task2.task2a import LU_decomposition
from common import create_table

Result = namedtuple('Result', ['matrix_dimension', 'condition_numbers_A', 'condition_numbers_L', 'condition_numbers_U'])


def experiment(dimension_range: Iterable, generate: Callable) -> List[Result]:
    results = []

    for dimension in dimension_range:
        A = np.array([[generate(row, column) for column in range(dimension)] for row in range(dimension)])
        decomposition = LU_decomposition(A)
        results.append(Result(dimension, calculate_condition_numbers(A), calculate_condition_numbers(decomposition.L),
                              calculate_condition_numbers(decomposition.U)))

    return results


def run():
    def unpack(experiment_results):
        return [(row.matrix_dimension,) + row.condition_numbers_A + row.condition_numbers_L + row.condition_numbers_U
                for row in experiment_results]

    dimension_range = range(2, 8)
    titles = ['matrix_dimension',
              'A_spectral_condition_number', 'A_volume_condition_number', 'A_angular_condition_number',
              'L_spectral_condition_number', 'L_volume_condition_number', 'L_angular_condition_number',
              'U_spectral_condition_number', 'U_volume_condition_number', 'U_angular_condition_number']

    create_table('results', 'task2a_hilberts_matrix_results.csv', titles,
                 unpack(experiment(dimension_range, generate_hilberts_matrix_element)))
    create_table('results', 'task2a_tridiagonal_matrix_results.csv', titles,
                 unpack(experiment(dimension_range, generate_tridiagonal_matrix_element)))


if __name__ == '__main__':
    run()
