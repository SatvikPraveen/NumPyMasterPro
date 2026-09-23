"""
Unit tests for linear_algebra_utils module
"""

import numpy as np
import pytest

from scripts.linear_algebra_utils import (
    PCAResult,
    back_substitution,
    cholesky_solve,
    closed_form_linear_regression,
    compute_determinant,
    compute_inverse,
    compute_rank,
    compute_svd,
    condition_number,
    dot_product,
    eigen_decomposition,
    forward_substitution,
    frobenius_norm,
    gram_schmidt,
    is_positive_definite,
    is_symmetric,
    l1_norm,
    l2_norm,
    least_squares,
    matmul_product,
    pca,
    power_iteration,
    print_matrix_with_header,
    project_onto,
    pseudo_inverse,
    qr_solve,
    ridge_regression,
    solve_system,
    transpose,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def spd_matrix(rng):
    A = rng.normal(size=(6, 6))
    return A @ A.T + 6 * np.eye(6)


class TestBasics:
    def test_products_and_transpose(self):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        assert dot_product(a, b) == 32
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(matmul_product(M, np.eye(2)), M)
        np.testing.assert_array_equal(transpose(M), M.T)

    def test_inverse_and_determinant(self):
        M = np.array([[4.0, 7.0], [2.0, 6.0]])
        np.testing.assert_allclose(compute_inverse(M) @ M, np.eye(2), atol=1e-12)
        assert compute_determinant(M) == pytest.approx(10.0)
        with pytest.raises(ValueError):
            compute_inverse(np.ones((2, 2)))
        with pytest.raises(ValueError):
            compute_inverse(np.ones((2, 3)))

    def test_rank_svd_pinv(self, rng):
        A = rng.normal(size=(5, 3))
        assert compute_rank(A) == 3
        assert compute_rank(np.ones((3, 3))) == 1
        U, S, Vt = compute_svd(A, full_matrices=False)
        np.testing.assert_allclose(U @ np.diag(S) @ Vt, A, atol=1e-12)
        np.testing.assert_allclose(pseudo_inverse(A) @ A, np.eye(3), atol=1e-10)

    def test_eigen(self):
        vals, vecs = eigen_decomposition(np.diag([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(sorted(vals), [1, 2, 3])
        assert vecs.shape == (3, 3)

    def test_norms(self):
        v = np.array([3.0, -4.0])
        assert l2_norm(v) == 5.0
        assert l1_norm(v) == 7.0
        assert frobenius_norm(np.array([[1.0, 2.0], [2.0, 1.0]])) == pytest.approx(np.sqrt(10))

    def test_solve_system(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        np.testing.assert_allclose(solve_system(A, b), [2.0, 3.0])
        with pytest.raises(ValueError):
            solve_system(np.ones((2, 2)), b)

    def test_project_onto(self):
        np.testing.assert_allclose(project_onto([3, 4], [1, 0]), [3, 0])
        with pytest.raises(ValueError):
            project_onto([1, 1], [0, 0])

    def test_print_helper(self, capsys):
        print_matrix_with_header(np.eye(2), "I")
        assert "I" in capsys.readouterr().out


class TestStructure:
    def test_condition_number(self):
        assert condition_number(np.eye(3)) == pytest.approx(1.0)
        assert condition_number(np.diag([1.0, 1e-8])) == pytest.approx(1e8)

    def test_symmetry_and_spd(self, spd_matrix):
        assert is_symmetric(spd_matrix)
        assert is_positive_definite(spd_matrix)
        assert not is_symmetric(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert not is_symmetric(np.ones((2, 3)))
        assert not is_positive_definite(-np.eye(2))
        assert not is_positive_definite(np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestTriangularSolvers:
    def test_forward_substitution(self, rng):
        L = np.tril(rng.normal(size=(5, 5))) + 5 * np.eye(5)
        b = rng.normal(size=5)
        np.testing.assert_allclose(forward_substitution(L, b), np.linalg.solve(L, b))

    def test_back_substitution(self, rng):
        U = np.triu(rng.normal(size=(5, 5))) + 5 * np.eye(5)
        b = rng.normal(size=5)
        np.testing.assert_allclose(back_substitution(U, b), np.linalg.solve(U, b))

    def test_only_triangle_is_read(self, rng):
        L = np.tril(rng.normal(size=(4, 4))) + 4 * np.eye(4)
        noisy = L + np.triu(rng.normal(size=(4, 4)), k=1)  # garbage above the diagonal
        b = rng.normal(size=4)
        np.testing.assert_allclose(forward_substitution(noisy, b), np.linalg.solve(L, b))

    def test_singular_diagonal(self):
        with pytest.raises(ValueError):
            forward_substitution(np.array([[0.0, 0.0], [1.0, 1.0]]), [1.0, 1.0])
        with pytest.raises(ValueError):
            back_substitution(np.array([[1.0, 1.0], [0.0, 0.0]]), [1.0, 1.0])

    def test_cholesky_solve(self, spd_matrix, rng):
        b = rng.normal(size=6)
        np.testing.assert_allclose(cholesky_solve(spd_matrix, b), np.linalg.solve(spd_matrix, b))
        with pytest.raises(ValueError):
            cholesky_solve(-np.eye(2), [1.0, 1.0])


class TestFactorisations:
    def test_gram_schmidt_reconstructs(self, rng):
        A = rng.normal(size=(8, 4))
        Q, R = gram_schmidt(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-12)
        np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-12)
        assert np.allclose(R, np.triu(R))

    def test_gram_schmidt_errors(self):
        with pytest.raises(ValueError):
            gram_schmidt(np.ones((2, 3)))
        with pytest.raises(ValueError):
            gram_schmidt(np.array([[1.0, 2.0], [2.0, 4.0]]))  # rank deficient

    def test_qr_solve_matches_lstsq(self, rng):
        A = rng.normal(size=(20, 3))
        b = rng.normal(size=20)
        np.testing.assert_allclose(qr_solve(A, b), np.linalg.lstsq(A, b, rcond=None)[0])


class TestSpectral:
    def test_power_iteration_dominant_eigenpair(self):
        A = np.diag([1.0, 5.0, 2.0])
        val, vec = power_iteration(A, seed=0)
        assert val == pytest.approx(5.0)
        np.testing.assert_allclose(np.abs(vec), [0, 1, 0], atol=1e-6)
        assert np.linalg.norm(vec) == pytest.approx(1.0)

    def test_power_iteration_symmetric_random(self, rng):
        M = rng.normal(size=(6, 6))
        A = M @ M.T
        val, vec = power_iteration(A, seed=1)
        assert val == pytest.approx(np.linalg.eigvalsh(A).max(), rel=1e-6)
        np.testing.assert_allclose(A @ vec, val * vec, atol=1e-6)

    def test_power_iteration_zero_matrix(self):
        val, vec = power_iteration(np.zeros((3, 3)), seed=0)
        assert val == 0.0
        assert np.linalg.norm(vec) == pytest.approx(1.0)


class TestPCA:
    def test_basic_properties(self, rng):
        X = rng.normal(size=(100, 5)) @ rng.normal(size=(5, 5))
        res = pca(X, n_components=3)
        assert isinstance(res, PCAResult)
        assert res.components.shape == (3, 5)
        assert res.transformed.shape == (100, 3)
        np.testing.assert_allclose(res.components @ res.components.T, np.eye(3), atol=1e-12)
        assert np.all(np.diff(res.explained_variance) <= 0)
        assert 0 < res.explained_variance_ratio.sum() <= 1.0 + 1e-12

    def test_full_reconstruction(self, rng):
        X = rng.normal(size=(30, 4))
        res = pca(X)
        np.testing.assert_allclose(res.explained_variance_ratio.sum(), 1.0)
        np.testing.assert_allclose(res.inverse_transform(res.transformed), X, atol=1e-10)

    def test_matches_eigendecomposition_of_covariance(self, rng):
        X = rng.normal(size=(50, 3))
        res = pca(X)
        cov_eigs = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
        np.testing.assert_allclose(res.explained_variance, cov_eigs)

    def test_finds_dominant_direction(self, rng):
        t = rng.normal(size=200)
        X = np.column_stack([t, 2 * t]) + 0.01 * rng.normal(size=(200, 2))
        res = pca(X, n_components=1)
        direction = res.components[0] / np.abs(res.components[0]).max()
        np.testing.assert_allclose(np.abs(direction), [0.5, 1.0], atol=0.02)
        assert res.explained_variance_ratio[0] > 0.99

    def test_deterministic_sign(self, rng):
        X = rng.normal(size=(20, 3))
        a, b = pca(X), pca(X.copy())
        np.testing.assert_array_equal(a.components, b.components)
        for comp in a.components:
            assert comp[np.argmax(np.abs(comp))] > 0

    def test_validation(self):
        with pytest.raises(ValueError):
            pca(np.ones((1, 3)))
        with pytest.raises(ValueError):
            pca(np.ones((5, 3)), n_components=4)
        with pytest.raises(ValueError):
            pca(np.arange(5))


class TestRegression:
    @pytest.fixture
    def linear_data(self, rng):
        X = rng.normal(size=(200, 3))
        true_w = np.array([1.5, -2.0, 0.5, 3.0])  # intercept first
        y = true_w[0] + X @ true_w[1:] + 0.01 * rng.normal(size=200)
        return X, y, true_w

    @pytest.mark.parametrize("method", ["lstsq", "qr", "cholesky", "normal"])
    def test_all_methods_agree(self, linear_data, method):
        X, y, true_w = linear_data
        w = closed_form_linear_regression(X, y, method=method)
        np.testing.assert_allclose(w, true_w, atol=0.01)

    def test_1d_feature_is_promoted(self, rng):
        x = np.linspace(0, 1, 50)
        y = 2 + 3 * x
        np.testing.assert_allclose(closed_form_linear_regression(x, y), [2, 3], atol=1e-10)

    def test_least_squares_rank_deficient_lstsq(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        b = np.array([2.0, 2.0, 2.0])
        w = least_squares(A, b, method="lstsq")
        np.testing.assert_allclose(A @ w, b)

    def test_least_squares_unknown_method(self):
        with pytest.raises(ValueError):
            least_squares(np.eye(2), [1, 1], method="bogus")  # type: ignore[arg-type]

    def test_ridge_shrinks_toward_zero(self, linear_data):
        X, y, _ = linear_data
        w0 = ridge_regression(X, y, alpha=0.0)
        w_big = ridge_regression(X, y, alpha=1e4)
        assert np.linalg.norm(w_big[1:]) < np.linalg.norm(w0[1:])
        # unpenalised intercept stays close to the mean of y under heavy shrinkage
        assert w_big[0] == pytest.approx(y.mean(), abs=0.1)

    def test_ridge_alpha_zero_equals_ols(self, linear_data):
        X, y, _ = linear_data
        np.testing.assert_allclose(
            ridge_regression(X, y, alpha=0.0), closed_form_linear_regression(X, y)
        )

    def test_ridge_no_intercept_closed_form(self, rng):
        X = rng.normal(size=(30, 2))
        y = rng.normal(size=30)
        alpha = 0.7
        expected = np.linalg.solve(X.T @ X + alpha * np.eye(2), X.T @ y)
        np.testing.assert_allclose(
            ridge_regression(X, y, alpha=alpha, fit_intercept=False), expected
        )

    def test_ridge_negative_alpha(self):
        with pytest.raises(ValueError):
            ridge_regression(np.eye(2), [1, 1], alpha=-1)
