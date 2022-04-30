import numpy as np

from practical_tasks.solutions.task5.common import MaxEigenValueAndVector


def find_using_power_method(A: np.ndarray, epsilon: float, max_iterations_count: int) -> MaxEigenValueAndVector:
    n = A.shape[0]

    Y, lambdas, iterations_count = np.ones(n), [1] * n, 0
    while True:
        previous_Y = Y
        Y = A.dot(previous_Y)
        lambdas = [Y[i] / previous_Y[i] for i in range(n)]
        iterations_count += 1

        # a posteriori estimate: |lambda - lambda_k| <= ||A * Y_k - lambda_k * Y_k|| / ||Y_k|| (< epsilon)
        if all([np.linalg.norm(A.dot(Y) - c * Y) / np.linalg.norm(Y) < epsilon for c in lambdas]):
            break

        if iterations_count > max_iterations_count:
            raise ValueError(f'Result not found after {max_iterations_count} iterations.')

    return MaxEigenValueAndVector(lambdas[0], Y, iterations_count)
