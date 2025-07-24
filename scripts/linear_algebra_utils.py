"""
linear_algebra_utils.py

A collection of linear algebra utility functions built on top of NumPy.
These are designed to simplify common linear algebra operations for educational
and production-grade numerical workflows.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

import numpy as np


# ✅ Dot & Matrix Products
def dot_product(a: np.ndarray, b: np.ndarray) -> np.number:
    """Computes the dot product of two vectors."""
    return np.dot(a, b)

def matmul_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs matrix multiplication (handles broadcasting)."""
    return np.matmul(a, b)


# ✅ Transposition
def transpose(matrix: np.ndarray) -> np.ndarray:
    """Returns the transpose of the given matrix."""
    return matrix.T


# ✅ Inverse, Determinant, and Rank
def compute_inverse(matrix: np.ndarray) -> np.ndarray:
    """Computes the inverse of a square matrix, if invertible."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not invertible.")

def compute_determinant(matrix: np.ndarray) -> float:
    """Returns the determinant of a square matrix."""
    return np.linalg.det(matrix)

def compute_rank(matrix: np.ndarray) -> int:
    """Computes the rank of a matrix."""
    return np.linalg.matrix_rank(matrix)


# ✅ Eigenvalues and Eigenvectors
def eigen_decomposition(matrix: np.ndarray) -> tuple:
    """Returns eigenvalues and eigenvectors of a square matrix."""
    return np.linalg.eig(matrix)


# ✅ Norms
def l2_norm(vector: np.ndarray) -> float:
    """Computes the L2 (Euclidean) norm of a vector."""
    return np.linalg.norm(vector)

def l1_norm(vector: np.ndarray) -> float:
    """Computes the L1 (Manhattan) norm of a vector."""
    return np.linalg.norm(vector, ord=1)


# ✅ Solving Linear Systems
def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves a linear system Ax = b for x.
    Raises an error if the matrix is singular or dimensions are incompatible.
    """
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("System cannot be solved: matrix is singular or inconsistent.")


# ✅ Singular Value Decomposition (SVD)
def compute_svd(matrix: np.ndarray) -> tuple:
    """
    Computes the Singular Value Decomposition of a matrix.
    Returns U, S, and V^T.
    """
    return np.linalg.svd(matrix)


# ✅ Pseudo-Inverse
def pseudo_inverse(matrix: np.ndarray) -> np.ndarray:
    """Returns the Moore-Penrose pseudo-inverse of a matrix."""
    return np.linalg.pinv(matrix)


# ✅ Optional: Pretty Print
def print_matrix_with_header(mat: np.ndarray, header: str = "Matrix") -> None:
    """Displays matrix with a visual header for clarity."""
    print(f"\n🧾 {header}:\n{mat}")
    