"""
linear_algebra_utils.py

Linear algebra utilities built on NumPy, for both educational and practical use.

Highlights
----------
- Least-squares regression through ``lstsq`` / QR / Cholesky instead of the
  numerically fragile ``inv(XᵀX)`` normal-equation shortcut
- Ridge regression with an un-penalised intercept
- Triangular solvers (forward / back substitution) written out explicitly
- Modified Gram-Schmidt QR, power iteration, and PCA via SVD
- Conditioning and structure checks (condition number, symmetry, SPD)

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
RandomState = int | np.random.Generator | None
RegressionMethod = Literal["lstsq", "qr", "cholesky", "normal"]

__all__ = [
    "PCAResult",
    "back_substitution",
    "cholesky_solve",
    "closed_form_linear_regression",
    "compute_determinant",
    "compute_inverse",
    "compute_rank",
    "compute_svd",
    "condition_number",
    "dot_product",
    "eigen_decomposition",
    "forward_substitution",
    "frobenius_norm",
    "gram_schmidt",
    "is_positive_definite",
    "is_symmetric",
    "l1_norm",
    "l2_norm",
    "least_squares",
    "matmul_product",
    "pca",
    "power_iteration",
    "print_matrix_with_header",
    "project_onto",
    "pseudo_inverse",
    "qr_solve",
    "ridge_regression",
    "solve_system",
    "transpose",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_matrix(a: npt.ArrayLike, name: str = "matrix") -> FloatArray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got shape {arr.shape}")
    return arr


def _as_square(a: npt.ArrayLike, name: str = "matrix") -> FloatArray:
    arr = _as_matrix(a, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square; got shape {arr.shape}")
    return arr


def _add_intercept(X: FloatArray) -> FloatArray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# Products, norms, basic decompositions (thin wrappers kept for the notebooks)
# ---------------------------------------------------------------------------
def dot_product(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Dot product of two arrays (``np.dot``)."""
    return np.dot(a, b)


def matmul_product(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Matrix product with broadcasting (``np.matmul``)."""
    return np.matmul(a, b)


def transpose(matrix: npt.ArrayLike) -> np.ndarray:
    """Transpose of the given matrix."""
    return np.asarray(matrix).T


def compute_inverse(matrix: npt.ArrayLike) -> FloatArray:
    """Inverse of a square matrix; raises ``ValueError`` if singular."""
    try:
        return np.linalg.inv(_as_square(matrix))
    except np.linalg.LinAlgError as err:
        raise ValueError("Matrix is not invertible.") from err


def compute_determinant(matrix: npt.ArrayLike) -> float:
    """Determinant of a square matrix."""
    return float(np.linalg.det(_as_square(matrix)))


def compute_rank(matrix: npt.ArrayLike) -> int:
    """Numerical rank via SVD."""
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float)))


def eigen_decomposition(matrix: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and right eigenvectors of a square matrix."""
    return np.linalg.eig(_as_square(matrix))


def l2_norm(vector: npt.ArrayLike) -> float:
    """Euclidean norm."""
    return float(np.linalg.norm(np.asarray(vector, dtype=float)))


def l1_norm(vector: npt.ArrayLike) -> float:
    """Manhattan norm."""
    return float(np.linalg.norm(np.asarray(vector, dtype=float).ravel(), ord=1))


def frobenius_norm(matrix: npt.ArrayLike) -> float:
    """Frobenius norm ``sqrt(Σ aᵢⱼ²)``."""
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord="fro"))


def solve_system(A: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """Solve ``Ax = b`` for square, non-singular ``A``."""
    try:
        return np.linalg.solve(_as_square(A, "A"), np.asarray(b, dtype=float))
    except np.linalg.LinAlgError as err:
        raise ValueError("System cannot be solved: matrix is singular or inconsistent.") from err


def compute_svd(
    matrix: npt.ArrayLike, full_matrices: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition ``U, S, Vᵀ``."""
    return np.linalg.svd(_as_matrix(matrix), full_matrices=full_matrices)


def pseudo_inverse(matrix: npt.ArrayLike) -> FloatArray:
    """Moore-Penrose pseudo-inverse."""
    return np.linalg.pinv(_as_matrix(matrix))


def print_matrix_with_header(mat: npt.ArrayLike, header: str = "Matrix") -> None:
    """Display a matrix with a visual header."""
    print(f"\n🧾 {header}:\n{np.asarray(mat)}")


def project_onto(v: npt.ArrayLike, u: npt.ArrayLike) -> FloatArray:
    """Orthogonal projection of ``v`` onto the direction ``u``: ``(v·u / u·u) u``."""
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    uu = float(u @ u)
    if uu == 0.0:
        raise ValueError("Cannot project onto the zero vector")
    return (float(v @ u) / uu) * u


# ---------------------------------------------------------------------------
# Structure & conditioning
# ---------------------------------------------------------------------------
def condition_number(
    matrix: npt.ArrayLike, p: Literal[1, -1, 2, -2, "fro", "nuc"] | None = 2
) -> float:
    """
    Condition number ``κ(A) = ||A|| · ||A⁻¹||`` (``np.linalg.cond``).

    Large values (≫ 1e8 in double precision) signal that solving with this
    matrix will amplify rounding error.
    """
    return float(np.linalg.cond(_as_matrix(matrix), p=p))


def is_symmetric(matrix: npt.ArrayLike, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    """True if ``A ≈ Aᵀ`` within tolerance (non-square inputs return False)."""
    a = _as_matrix(matrix)
    return a.shape[0] == a.shape[1] and bool(np.allclose(a, a.T, rtol=rtol, atol=atol))


def is_positive_definite(matrix: npt.ArrayLike) -> bool:
    """True if ``A`` is symmetric positive definite (Cholesky succeeds)."""
    a = _as_matrix(matrix)
    if not is_symmetric(a):
        return False
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        return False
    return True


# ---------------------------------------------------------------------------
# Triangular solvers & factorisations
# ---------------------------------------------------------------------------
def forward_substitution(L: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """
    Solve ``L x = b`` for lower-triangular ``L`` (row by row, vectorised per row).

    Only the lower triangle of ``L`` is read.
    """
    L = _as_square(L, "L")
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    if np.any(np.diag(L) == 0):
        raise ValueError("L has a zero on its diagonal; system is singular")
    x = np.zeros_like(b, dtype=float)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def back_substitution(U: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """
    Solve ``U x = b`` for upper-triangular ``U`` (bottom-up, vectorised per row).

    Only the upper triangle of ``U`` is read.
    """
    U = _as_square(U, "U")
    b = np.asarray(b, dtype=float)
    n = U.shape[0]
    if np.any(np.diag(U) == 0):
        raise ValueError("U has a zero on its diagonal; system is singular")
    x = np.zeros_like(b, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def cholesky_solve(A: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """
    Solve ``A x = b`` for symmetric positive-definite ``A`` via ``A = L Lᵀ``.

    Two triangular solves: ``L y = b`` then ``Lᵀ x = y``. About twice as fast
    as LU for SPD systems and numerically very stable.
    """
    A = _as_square(A, "A")
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as err:
        raise ValueError("A must be symmetric positive definite for cholesky_solve") from err
    y = forward_substitution(L, b)
    return back_substitution(L.T, y)


def gram_schmidt(A: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """
    Thin QR factorisation ``A = Q R`` by *modified* Gram-Schmidt.

    Modified GS re-orthogonalises against the already-computed ``q`` vectors one
    at a time, which is far more stable than classical GS in floating point.

    Parameters
    ----------
    A : array_like, shape (m, n) with ``m >= n`` and full column rank

    Returns
    -------
    Q : (m, n) with orthonormal columns
    R : (n, n) upper triangular
    """
    A = _as_matrix(A, "A")
    m, n = A.shape
    if m < n:
        raise ValueError("gram_schmidt requires m >= n (tall or square matrix)")
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] < 1e-12:
            raise ValueError(f"Matrix is rank deficient; column {j} is linearly dependent")
        Q[:, j] = V[:, j] / R[j, j]
        # Remove the q_j component from every remaining column at once.
        R[j, j + 1 :] = Q[:, j] @ V[:, j + 1 :]
        V[:, j + 1 :] -= np.outer(Q[:, j], R[j, j + 1 :])
    return Q, R


def qr_solve(A: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """
    Least-squares solution of ``A x ≈ b`` via Householder QR.

    ``min ||Ax - b||₂`` is solved as ``R x = Qᵀ b`` using back substitution;
    unlike the normal equations this never squares the condition number.
    """
    A = _as_matrix(A, "A")
    b = np.asarray(b, dtype=float)
    Q, R = np.linalg.qr(A, mode="reduced")
    return back_substitution(R, Q.T @ b)


# ---------------------------------------------------------------------------
# Eigen / spectral
# ---------------------------------------------------------------------------
def power_iteration(
    A: npt.ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-10,
    seed: RandomState = None,
) -> tuple[float, FloatArray]:
    """
    Dominant eigenpair of a square matrix by power iteration.

    Repeatedly applies ``A`` to a random unit vector; converges to the
    eigenvector of the eigenvalue with largest magnitude (assuming it is unique).

    Returns
    -------
    (eigenvalue, eigenvector) with ``||eigenvector|| = 1``
    """
    A = _as_square(A, "A")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    eigenvalue = 0.0
    for _ in range(max_iter):
        w = A @ v
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return 0.0, v
        v_new = w / norm
        eigenvalue = float(v_new @ A @ v_new)  # Rayleigh quotient
        if np.linalg.norm(v_new - v) < tol or np.linalg.norm(v_new + v) < tol:
            v = v_new
            break
        v = v_new
    return eigenvalue, v


@dataclass(frozen=True)
class PCAResult:
    """Output of :func:`pca`."""

    components: FloatArray  # (n_components, n_features), rows are principal axes
    explained_variance: FloatArray  # (n_components,)
    explained_variance_ratio: FloatArray  # (n_components,)
    mean: FloatArray  # (n_features,)
    transformed: FloatArray  # (n_samples, n_components)

    def inverse_transform(self, Z: npt.ArrayLike) -> FloatArray:
        """Map scores back to the original feature space."""
        return np.asarray(Z, dtype=float) @ self.components + self.mean


def pca(X: npt.ArrayLike, n_components: int | None = None) -> PCAResult:
    """
    Principal component analysis via SVD of the centred data.

    With ``X_c = X - mean`` and ``X_c = U S Vᵀ``, the principal axes are the rows
    of ``Vᵀ``, the scores are ``U S``, and the explained variance of component
    *i* is ``sᵢ² / (n - 1)``. Signs are normalised so the largest-magnitude
    entry of each component is positive (deterministic output).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    n_components : int, optional
        Defaults to ``min(n_samples, n_features)``.
    """
    X = _as_matrix(X, "X")
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("PCA needs at least 2 samples")
    max_comp = min(n_samples, n_features)
    k = max_comp if n_components is None else n_components
    if not 1 <= k <= max_comp:
        raise ValueError(f"n_components must be in [1, {max_comp}]")

    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Deterministic sign convention.
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    signs[signs == 0] = 1.0
    Vt *= signs[:, None]
    U *= signs[None, :]

    explained = S**2 / (n_samples - 1)
    total_var = float(np.sum(Xc**2)) / (n_samples - 1)
    ratio = explained / total_var if total_var > 0 else np.zeros_like(explained)

    return PCAResult(
        components=Vt[:k],
        explained_variance=explained[:k],
        explained_variance_ratio=ratio[:k],
        mean=mean,
        transformed=(U[:, :k] * S[:k]),
    )


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def least_squares(
    A: npt.ArrayLike, b: npt.ArrayLike, method: RegressionMethod = "lstsq"
) -> FloatArray:
    """
    Solve ``min ||A x - b||₂`` with a choice of algorithm.

    ``"lstsq"``    SVD-based (``np.linalg.lstsq``); handles rank deficiency.
    ``"qr"``       Householder QR + back substitution.
    ``"cholesky"`` Normal equations solved with Cholesky (fast; needs full column rank).
    ``"normal"``   Explicit ``inv(AᵀA) Aᵀ b`` — included for comparison only.
    """
    A = _as_matrix(A, "A")
    b = np.asarray(b, dtype=float)
    if method == "lstsq":
        return np.linalg.lstsq(A, b, rcond=None)[0]
    if method == "qr":
        return qr_solve(A, b)
    if method == "cholesky":
        return cholesky_solve(A.T @ A, A.T @ b)
    if method == "normal":
        return np.linalg.inv(A.T @ A) @ A.T @ b
    raise ValueError(f"Unknown method {method!r}")


def closed_form_linear_regression(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    method: RegressionMethod = "lstsq",
) -> FloatArray:
    """
    Ordinary least squares with an intercept term.

    Returns ``w`` of length ``n_features + 1`` with ``w[0]`` the intercept, so
    predictions are ``X_b @ w`` where ``X_b = [1, X]``. The default solver is
    SVD-based ``lstsq``; the historical ``inv(XᵀX) Xᵀ y`` formula is available
    as ``method="normal"``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return least_squares(_add_intercept(X), np.asarray(y, dtype=float), method=method)


def ridge_regression(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    alpha: float = 1.0,
    fit_intercept: bool = True,
) -> FloatArray:
    """
    Ridge (L2-regularised) regression: ``min ||Xw - y||² + α||w||²``.

    Solves ``(XᵀX + αI) w = Xᵀy`` with Cholesky. When ``fit_intercept`` is
    True the intercept is *not* penalised, matching scikit-learn's behaviour.

    Returns
    -------
    w : ndarray
        ``[intercept, coef...]`` if ``fit_intercept`` else ``coef``.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        Xb = _add_intercept(X)
        penalty = alpha * np.eye(Xb.shape[1])
        penalty[0, 0] = 0.0
    else:
        Xb = X
        penalty = alpha * np.eye(Xb.shape[1])
    lhs = Xb.T @ Xb + penalty
    rhs = Xb.T @ y
    if alpha == 0.0:
        return np.linalg.lstsq(Xb, y, rcond=None)[0]
    return cholesky_solve(lhs, rhs)
