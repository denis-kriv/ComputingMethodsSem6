import random


def generate_hilberts_matrix_element(row, column):
    return 1 / (row + column + 1)


def generate_tridiagonal_matrix_element(row, column):
    return random.randint(-1000, 1000) if abs(row - column) <= 1 else 0


def generate_diagonal_dominance_matrix_element(row, column):
    return random.randint(1000, 10000) if row == column else random.randint(-10, 10)
