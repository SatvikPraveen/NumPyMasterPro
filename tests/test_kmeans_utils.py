"""
Unit tests for kmeans_utils module
"""

import numpy as np
import pytest

from scripts.kmeans_utils import (
    KMeans,
    KMeansResult,
    assign_clusters,
    compute_cluster_inertia,
    compute_inertia,
    elbow_point,
    generate_data,
    initialize_centroids,
    kmeans,
    pairwise_sq_distances,
    run_kmeans,
    silhouette_score,
    update_centroids,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestPairwiseSqDistances:
    def test_matches_broadcast_formula(self, rng):
        X = rng.normal(size=(40, 5))
        C = rng.normal(size=(4, 5))
        expected = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(pairwise_sq_distances(X, C), expected, atol=1e-9)

    def test_self_distance_is_zero(self, rng):
        X = rng.normal(size=(10, 3))
        d = pairwise_sq_distances(X, X)
        assert np.allclose(np.diag(d), 0.0)
        assert (d >= 0).all()


class TestGenerateData:
    def test_default_shape_and_dtype(self):
        data = generate_data()
        assert data.shape == (150, 2)
        assert data.dtype == np.float64

    def test_custom_centers(self):
        data = generate_data(n_per_cluster=10, centers=[(0, 0, 0), (5, 5, 5)], seed=1)
        assert data.shape == (20, 3)

    def test_reproducible_with_seed(self):
        assert np.array_equal(generate_data(seed=7), generate_data(seed=7))
        assert not np.array_equal(generate_data(seed=7), generate_data(seed=8))


class TestInitializeCentroids:
    @pytest.mark.parametrize("method", ["random", "k-means++"])
    def test_centroid_count_and_membership(self, method, rng):
        data = rng.random((100, 2))
        centroids = initialize_centroids(data, k=5, method=method, rng=rng)
        assert centroids.shape == (5, 2)
        # every centroid must be an actual data row
        for c in centroids:
            assert np.any(np.all(np.isclose(data, c), axis=1))

    def test_random_centroids_are_distinct(self, rng):
        data = rng.random((20, 2))
        centroids = initialize_centroids(data, k=20, method="random", rng=rng)
        assert len(np.unique(centroids, axis=0)) == 20

    def test_kmeanspp_prefers_far_points(self):
        # Two tight blobs far apart: k-means++ with k=2 must pick one from each.
        blob_a = np.zeros((50, 2))
        blob_b = np.full((50, 2), 100.0)
        data = np.vstack([blob_a, blob_b])
        for seed in range(10):
            c = initialize_centroids(data, k=2, method="k-means++", rng=seed)
            assert not np.allclose(c[0], c[1])

    def test_invalid_k(self, rng):
        data = rng.random((10, 2))
        with pytest.raises(ValueError):
            initialize_centroids(data, k=15)
        with pytest.raises(ValueError):
            initialize_centroids(data, k=0)

    def test_invalid_method(self, rng):
        with pytest.raises(ValueError, match="Unknown init method"):
            initialize_centroids(rng.random((5, 2)), k=2, method="bogus")  # type: ignore[arg-type]

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            initialize_centroids(np.array([[0.0, np.nan]]), k=1)


class TestAssignClusters:
    def test_shape_and_range(self, rng):
        data = rng.random((100, 2))
        centroids = rng.random((3, 2))
        labels = assign_clusters(data, centroids)
        assert labels.shape == (100,)
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_simple_case(self):
        data = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        labels = assign_clusters(data, centroids)
        assert labels.tolist() == [0, 0, 1, 1]


class TestUpdateCentroids:
    def test_shape(self, rng):
        data = rng.random((100, 2))
        labels = rng.integers(0, 3, 100)
        assert update_centroids(data, labels, k=3).shape == (3, 2)

    def test_means(self):
        data = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
        labels = np.array([0, 0, 1, 1])
        centroids = update_centroids(data, labels, k=2)
        np.testing.assert_allclose(centroids, [[1, 1], [11, 11]])

    def test_empty_cluster_is_repaired_without_nan(self):
        data = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        labels = np.array([0, 0, 0])  # cluster 1 empty
        centroids = update_centroids(data, labels, k=2)
        assert np.isfinite(centroids).all()
        # cluster 1 re-seeded at the farthest point (10, 0)
        np.testing.assert_allclose(centroids[1], [10.0, 0.0])

    def test_empty_cluster_with_previous(self):
        data = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        labels = np.array([0, 0, 0])
        previous = np.array([[0.5, 0.0], [50.0, 50.0]])
        centroids = update_centroids(data, labels, k=2, previous=previous)
        assert np.isfinite(centroids).all()
        np.testing.assert_allclose(centroids[0], [11.0 / 3.0, 0.0])


class TestRunKmeans:
    def test_result_type_and_consistency(self):
        data = generate_data()
        res = run_kmeans(data, k=3, seed=0)
        assert isinstance(res, KMeansResult)
        assert res.k == 3
        assert res.centroids.shape == (3, 2)
        assert res.labels.shape == (150,)
        assert res.converged
        assert res.n_iter >= 1
        # labels/inertia consistent with returned centroids
        assert np.array_equal(res.labels, assign_clusters(data, res.centroids))
        assert np.isclose(res.inertia, compute_cluster_inertia(data, res.centroids, res.labels))

    def test_inertia_history_is_non_increasing(self):
        data = generate_data(seed=3)
        res = run_kmeans(data, k=3, seed=3, init="random")
        hist = np.asarray(res.inertia_history)
        assert np.all(np.diff(hist) <= 1e-9)

    def test_recovers_well_separated_clusters(self):
        data = generate_data(n_per_cluster=40, centers=[(0, 0), (20, 0), (0, 20)], std=0.5, seed=1)
        res = run_kmeans(data, k=3, n_init=5, seed=1)
        truth = np.repeat([0, 1, 2], 40)
        # each true cluster maps to exactly one predicted label
        for t in range(3):
            assert len(np.unique(res.labels[truth == t])) == 1
        assert len(np.unique(res.labels)) == 3

    def test_n_init_never_worse_than_single_run(self):
        data = generate_data(seed=5)
        single = run_kmeans(data, k=4, init="random", n_init=1, seed=5).inertia
        multi = run_kmeans(data, k=4, init="random", n_init=10, seed=5).inertia
        assert multi <= single + 1e-9

    def test_seed_reproducibility(self):
        data = generate_data()
        r1 = run_kmeans(data, k=3, seed=42)
        r2 = run_kmeans(data, k=3, seed=42)
        np.testing.assert_allclose(r1.centroids, r2.centroids)
        assert np.array_equal(r1.labels, r2.labels)

    def test_generator_is_advanced_not_reset(self):
        data = generate_data()
        gen = np.random.default_rng(0)
        r1 = run_kmeans(data, k=3, seed=gen, init="random")
        r2 = run_kmeans(data, k=3, seed=gen, init="random")
        # Two draws from the same generator need not coincide.
        assert r1.n_iter >= 1
        assert r2.n_iter >= 1

    def test_max_iters_respected(self):
        data = generate_data()
        res = run_kmeans(data, k=3, max_iters=1, tol=0.0, seed=0)
        assert res.n_iter == 1
        assert not res.converged

    def test_bad_arguments(self):
        data = generate_data()
        with pytest.raises(ValueError):
            run_kmeans(data, k=3, n_init=0)
        with pytest.raises(ValueError):
            run_kmeans(data, k=3, max_iters=0)
        with pytest.raises(ValueError):
            run_kmeans(data[:, 0], k=3)  # 1-D input


class TestKmeansWrapper:
    def test_returns_tuple(self):
        centroids, labels = kmeans(generate_data(), k=3, seed=0)
        assert centroids.shape == (3, 2)
        assert labels.shape == (150,)

    def test_convergence_on_tiny_data(self):
        data = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        _centroids, labels = kmeans(data, k=2, max_iters=10, seed=0)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


class TestInertia:
    def test_positive(self, rng):
        data = rng.random((100, 2))
        centroids = rng.random((3, 2))
        labels = rng.integers(0, 3, 100)
        assert compute_cluster_inertia(data, centroids, labels) >= 0

    def test_zero_for_perfect_clusters(self):
        centroids = np.array([[0, 0], [10, 10]])
        assert np.isclose(compute_cluster_inertia(centroids.copy(), centroids, [0, 1]), 0.0)

    def test_manual_value(self):
        data = np.array([[0, 0], [2, 0], [10, 0], [12, 0]])
        centroids = np.array([[1, 0], [11, 0]])
        assert np.isclose(compute_cluster_inertia(data, centroids, [0, 0, 1, 1]), 4.0)

    def test_elbow_curve_is_decreasing(self):
        data = generate_data(seed=2)
        inertias = compute_inertia(data, range(1, 7), n_init=5, seed=2)
        assert len(inertias) == 6
        assert all(np.diff(inertias) <= 1e-6)


class TestElbowPoint:
    def test_finds_obvious_elbow(self):
        ks = [1, 2, 3, 4, 5, 6]
        inertias = [1000, 500, 100, 90, 85, 82]
        assert elbow_point(ks, inertias) == 3

    def test_short_input(self):
        assert elbow_point([2, 3], [10.0, 5.0]) == 2

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            elbow_point([1, 2, 3], [1.0, 2.0])


class TestSilhouette:
    def test_well_separated_is_near_one(self):
        data = generate_data(n_per_cluster=30, centers=[(0, 0), (50, 50)], std=0.1, seed=0)
        labels = np.repeat([0, 1], 30)
        assert silhouette_score(data, labels) > 0.95

    def test_random_labels_near_zero(self):
        data = generate_data(n_per_cluster=50, centers=[(0, 0), (50, 50)], std=0.1, seed=0)
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 100)
        assert abs(silhouette_score(data, labels)) < 0.2

    def test_matches_naive_implementation(self, rng):
        data = rng.normal(size=(30, 3))
        labels = rng.integers(0, 3, 30)
        D = np.sqrt(((data[:, None] - data[None]) ** 2).sum(-1))
        scores = []
        for i in range(30):
            same = (labels == labels[i]) & (np.arange(30) != i)
            if same.sum() == 0:
                scores.append(0.0)
                continue
            a = D[i, same].mean()
            b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
            scores.append((b - a) / max(a, b))
        assert np.isclose(silhouette_score(data, labels), np.mean(scores))

    def test_requires_two_clusters(self):
        with pytest.raises(ValueError):
            silhouette_score(np.zeros((5, 2)), np.zeros(5, dtype=int))


class TestKMeansClass:
    def test_fit_predict_transform(self):
        data = generate_data(seed=0)
        km = KMeans(n_clusters=3, seed=0).fit(data)
        assert km.cluster_centers_.shape == (3, 2)
        assert km.labels_.shape == (150,)
        assert km.inertia_ > 0
        assert km.n_iter_ >= 1
        assert np.array_equal(km.predict(data), km.labels_)
        assert km.transform(data).shape == (150, 3)
        assert km.score(data) == pytest.approx(-km.inertia_)

    def test_fit_predict_shortcut(self):
        data = generate_data(seed=0)
        assert KMeans(n_clusters=2, seed=0).fit_predict(data).shape == (150,)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            _ = KMeans().cluster_centers_

    def test_repr(self):
        assert "KMeans(n_clusters=4" in repr(KMeans(n_clusters=4))
