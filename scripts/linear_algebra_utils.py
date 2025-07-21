import numpy as np

def solve_system(A, b):
    return np.linalg.solve(A, b)

def compute_inverse(matrix):
    return np.linalg.inv(matrix)

def dot_product(a, b):
    return np.dot(a, b)