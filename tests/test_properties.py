"""
Property-based tests (hypothesis) for invariants that should hold for *any* input,
not just the hand-picked examples in the unit tests.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from scripts.array_utils import broadcast_result_shape
from scripts.kmeans_utils import (
    KMeans,
    initialize_centroids,
    pairwise_sq_distances,
    run_kmeans,
    update_centroids,
)
from scripts.linear_algebra_utils import (
    back_substitution,
    cholesky_solve,
    forward_substitution,
    gram_schmidt,
    least_squares,
    pca,
)
from scripts.stats_utils import (
    RunningStats,
    entropy,
    minmax_normalize,
    moving_average,
    pearson_r,
    zscore_normalize,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
finite_floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)

vectors = hnp.arrays(
    np.float64,
    hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=50),
    elements=finite_floats,
)

matrices = hnp.arrays(
    np.float64,
    hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=12),
    elements=finite_floats,
)

# (n_samples, n_features) data with n_samples >= 3 so k in [2, n) is possible
datasets = hnp.arrays(
    np.float64,
    st.tuples(st.integers(3, 40), st.integers(1, 5)),
    elements=finite_floats,
)


slow = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
class TestKMeansProperties:
    @slow
    @given(X=datasets, C=datasets)
    def test_pairwise_sq_distances_matches_broadcast(self, X, C):
        if X.shape[1] != C.shape[1]:
            C = (
                C[:, : X.shape[1]]
                if C.shape[1] > X.shape[1]
                else np.pad(C, ((0, 0), (0, X.shape[1] - C.shape[1])))
            )
        expected = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        got = pairwise_sq_distances(X, C)
        scale = max(1.0, float(np.abs(expected).max()))
        np.testing.assert_allclose(got, expected, atol=1e-6 * scale, rtol=1e-6)
        assert (got >= 0).all()

    @slow
    @given(
        X=datasets, seed=st.integers(0, 2**31 - 1), method=st.sampled_from(["random", "k-means++"])
    )
    def test_initial_centroids_are_data_rows(self, X, seed, method):
        k = min(3, X.shape[0])
        C = initialize_centroids(X, k, method=method, rng=seed)
        assert C.shape == (k, X.shape[1])
        for c in C:
            assert np.any(np.all(c == X, axis=1))

    @slow
    @given(X=datasets, seed=st.integers(0, 2**31 - 1))
    def test_run_kmeans_invariants(self, X, seed):
        k = 2
        res = run_kmeans(X, k=k, seed=seed, n_init=2)
        assert res.labels.shape == (X.shape[0],)
        assert res.labels.min() >= 0
        assert res.labels.max() < k
        assert np.isfinite(res.centroids).all()
        assert res.inertia >= 0
        hist = np.asarray(res.inertia_history)
        scale = max(1.0, float(hist.max()))
        assert np.all(np.diff(hist) <= 1e-9 * scale), (
            "inertia must not increase across Lloyd iterations"
        )

    @slow
    @given(X=datasets, seed=st.integers(0, 2**31 - 1))
    def test_update_centroids_never_nan(self, X, seed):
        rng = np.random.default_rng(seed)
        k = 4
        labels = rng.integers(0, 2, size=X.shape[0])  # clusters 2 and 3 always empty
        C = update_centroids(X, labels, k)
        assert C.shape == (k, X.shape[1])
        assert np.isfinite(C).all()

    @slow
    @given(X=datasets, seed=st.integers(0, 2**31 - 1))
    def test_estimator_predict_is_consistent_with_labels(self, X, seed):
        km = KMeans(n_clusters=2, seed=seed, n_init=1).fit(X)
        assert np.array_equal(km.predict(X), km.labels_)
        assert km.transform(X).shape == (X.shape[0], 2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class TestStatsProperties:
    @slow
    @given(x=vectors)
    def test_minmax_in_unit_interval(self, x):
        out = minmax_normalize(x)
        assert np.isfinite(out).all()
        assert out.min() >= -1e-12
        assert out.max() <= 1 + 1e-12
        if np.ptp(x) > 0:
            assert out.min() == pytest.approx(0.0)
            assert out.max() == pytest.approx(1.0)

    @slow
    @given(x=vectors)
    def test_zscore_moments(self, x):
        z = zscore_normalize(x)
        assert np.isfinite(z).all()
        if np.std(x) > 1e-6 * max(1.0, np.abs(x).max()):
            assert z.mean() == pytest.approx(0.0, abs=1e-6)
            assert z.std() == pytest.approx(1.0, abs=1e-6)
        elif np.ptp(x) == 0:
            assert np.allclose(z, 0.0)

    @slow
    @given(x=vectors, n_chunks=st.integers(1, 6))
    def test_running_stats_equals_numpy(self, x, n_chunks):
        rs = RunningStats()
        for chunk in np.array_split(x, min(n_chunks, x.size)):
            rs.update(chunk)
        assert rs.n == x.size
        assert rs.mean == pytest.approx(x.mean(), abs=1e-9, rel=1e-9)
        assert rs.variance() == pytest.approx(x.var(), abs=1e-6, rel=1e-6)
        assert rs.min == x.min()
        assert rs.max == x.max()

    @slow
    @given(x=vectors, data=st.data())
    def test_moving_average_bounded_by_window_extrema(self, x, data):
        window = data.draw(st.integers(1, x.size))
        out = moving_average(x, window)
        assert out.shape == (x.size - window + 1,)
        assert out.min() >= x.min() - 1e-9
        assert out.max() <= x.max() + 1e-9

    @slow
    @given(
        counts=hnp.arrays(
            np.float64, st.integers(1, 20), elements=st.floats(0, 1e3, allow_nan=False)
        )
    )
    def test_entropy_bounds(self, counts):
        if counts.sum() == 0:
            return
        h = entropy(counts)
        assert np.isfinite(h)
        assert -1e-12 <= h <= np.log(counts.size) + 1e-9

    @slow
    @given(x=vectors)
    def test_pearson_r_in_range_and_self_correlation(self, x):
        assert -1 - 1e-9 <= pearson_r(x, x[::-1]) <= 1 + 1e-9
        if np.std(x) > 1e-6 * max(1.0, np.abs(x).max()):
            assert pearson_r(x, x) == pytest.approx(1.0, abs=1e-6)
            assert pearson_r(x, -x) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def _well_conditioned_square(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(size=(n, n)) + n * np.eye(n)


class TestLinalgProperties:
    @slow
    @given(n=st.integers(1, 10), seed=st.integers(0, 2**31 - 1))
    def test_triangular_solvers_match_numpy(self, n, seed):
        rng = np.random.default_rng(seed)
        A = _well_conditioned_square(rng, n)
        b = rng.normal(size=n)
        L, U = np.tril(A), np.triu(A)
        np.testing.assert_allclose(
            forward_substitution(L, b), np.linalg.solve(L, b), rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(
            back_substitution(U, b), np.linalg.solve(U, b), rtol=1e-8, atol=1e-8
        )

    @slow
    @given(n=st.integers(1, 10), seed=st.integers(0, 2**31 - 1))
    def test_cholesky_solve_matches_numpy(self, n, seed):
        rng = np.random.default_rng(seed)
        M = rng.normal(size=(n, n))
        A = M @ M.T + n * np.eye(n)
        b = rng.normal(size=n)
        np.testing.assert_allclose(
            cholesky_solve(A, b), np.linalg.solve(A, b), rtol=1e-7, atol=1e-8
        )

    @slow
    @given(m=st.integers(2, 12), n=st.integers(1, 6), seed=st.integers(0, 2**31 - 1))
    def test_gram_schmidt_reconstruction(self, m, n, seed):
        n = min(n, m)
        A = np.random.default_rng(seed).normal(size=(m, n))
        Q, R = gram_schmidt(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-9)
        np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-9)
        assert np.allclose(R, np.triu(R))

    @slow
    @given(m=st.integers(3, 20), n=st.integers(1, 3), seed=st.integers(0, 2**31 - 1))
    def test_least_squares_methods_agree(self, m, n, seed):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(m, n))
        b = rng.normal(size=m)
        ref = least_squares(A, b, method="lstsq")
        for method in ("qr", "cholesky"):
            np.testing.assert_allclose(
                least_squares(A, b, method=method), ref, rtol=1e-6, atol=1e-8
            )

    @slow
    @given(m=st.integers(2, 20), n=st.integers(1, 5), seed=st.integers(0, 2**31 - 1))
    def test_pca_round_trip_and_variance_budget(self, m, n, seed):
        X = np.random.default_rng(seed).normal(size=(m, n))
        res = pca(X)
        np.testing.assert_allclose(res.inverse_transform(res.transformed), X, atol=1e-9)
        assert res.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(res.explained_variance) <= 1e-12)


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------
class TestBroadcastProperties:
    @slow
    @given(
        shapes=st.lists(
            st.lists(st.sampled_from([1, 2, 3, 5]), max_size=4).map(tuple), min_size=1, max_size=4
        )
    )
    def test_matches_numpy_broadcast_shapes(self, shapes):
        try:
            expected = np.broadcast_shapes(*shapes)
        except ValueError:
            with pytest.raises(ValueError):
                broadcast_result_shape(*shapes)
        else:
            assert broadcast_result_shape(*shapes) == expected
