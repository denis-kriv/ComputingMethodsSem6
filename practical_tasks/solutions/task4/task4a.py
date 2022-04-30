import numpy as np

from practical_tasks.solutions.task4.common import SolutionUsingIterativeMethod, solve_using_iterative_method


# A with diagonal dominance or A = A* and lambda > 0 for all lambda -- eigen value of A
def solve_using_simple_iterations_method(A: np.ndarray, b: np.ndarray, epsilon: float) -> SolutionUsingIterativeMethod:
    n = A.shape[0]

    if all([2 * abs(A[row, row]) > sum([abs(A[row, column]) for column in range(n)]) for row in range(n)]):
        B, c = map(np.array, zip(*[([0 if row == column else -A[row, column] / A[row, row] for column in range(n)],
                                    b[row] / A[row, row]) for row in range(n)]))
    else:
        eigen_values = np.linalg.eigvals(A)
        if any([v <= 0 for v in eigen_values]) or not np.allclose(A.T, np.conjugate(A)):
            raise ValueError('A without diagonal dominance and A != A*.')
        alpha = 2 / (min(eigen_values) + max(eigen_values))
        B, c = np.eye(n) - alpha * A, alpha * b

    return solve_using_iterative_method(B, c, epsilon)
