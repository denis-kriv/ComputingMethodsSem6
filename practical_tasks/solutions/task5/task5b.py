import numpy as np

from practical_tasks.solutions.task5.common import MaxEigenValueAndVector


# A -- symmetric
def find_using_scalar_product_method(A: np.ndarray, epsilon: float,
                                     max_iterations_count: int) -> MaxEigenValueAndVector:
    if not np.allclose(A, A.T):
        raise ValueError('A is not symmetric.')

    n = A.shape[0]

    Y, lambdas, iterations_count = np.ones(n), 0, 0
    while True:
        previous_Y = Y
        Y = A.dot(previous_Y)
        lambdas = previous_Y.dot(Y) / previous_Y.dot(previous_Y)
        iterations_count += 1

        # a posteriori estimate: |lambda - lambda_k| <= ||A * Y_k - lambda_k * Y_k|| / ||Y_k|| (< epsilon)
        if np.linalg.norm(A.dot(Y) - lambdas * Y) / np.linalg.norm(Y) < epsilon:
            break

        if iterations_count > max_iterations_count:
            raise ValueError(f'Result not found after {max_iterations_count} iterations.')

    return MaxEigenValueAndVector(lambdas, Y, iterations_count)
