"""
Unit tests for stats_utils module
"""

import numpy as np
import pytest

from scripts.stats_utils import (
    RunningStats,
    bootstrap_ci,
    compute_bincount,
    compute_correlation,
    compute_covariance,
    covariance_matrix,
    ecdf,
    entropy,
    generate_normal_distribution,
    generate_random_floats,
    generate_random_integers,
    histogram_binning,
    minmax_normalize,
    moving_average,
    pearson_r,
    quantiles,
    random_sample,
    robust_scale,
    standardize_rows,
    summarize_array,
    weighted_mean,
    weighted_std,
    weighted_var,
    zscore_normalize,
)


class TestSummarizeArray:
    def test_basic_values(self):
        s = summarize_array([1, 2, 3, 4, 5])
        assert s["count"] == 5
        assert s["mean"] == 3
        assert s["median"] == 3
        assert s["min"] == 1
        assert s["max"] == 5
        assert s["range"] == 4
        assert s["IQR"] == 2
        assert s["skewness"] == pytest.approx(0.0)

    def test_skew_and_kurtosis_of_normal_sample(self):
        x = np.random.default_rng(0).normal(size=200_000)
        s = summarize_array(x)
        assert abs(s["skewness"]) < 0.05
        assert abs(s["kurtosis"]) < 0.05

    def test_constant_array_has_zero_moments(self):
        s = summarize_array(np.full(10, 3.0))
        assert s["std"] == 0
        assert s["skewness"] == 0
        assert s["kurtosis"] == 0

    def test_nan_policies(self):
        x = np.array([1.0, np.nan, 3.0])
        assert np.isnan(summarize_array(x)["mean"])
        assert summarize_array(x, nan_policy="omit")["mean"] == 2.0
        assert summarize_array(x, nan_policy="omit")["count"] == 2
        with pytest.raises(ValueError):
            summarize_array(x, nan_policy="raise")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summarize_array([])


class TestCorrelationCovariance:
    def test_quantiles(self):
        np.testing.assert_allclose(quantiles(np.arange(1, 6)), [2, 3, 4])

    def test_np_wrappers(self):
        x, y = np.array([1.0, 2, 3, 4]), np.array([2.0, 4, 6, 8])
        assert compute_covariance(x, y).shape == (2, 2)
        assert compute_correlation(x, y)[0, 1] == pytest.approx(1.0)

    def test_pearson_r_matches_numpy(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=50), rng.normal(size=50)
        assert pearson_r(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])

    def test_pearson_r_constant_input(self):
        assert pearson_r([1, 1, 1], [1, 2, 3]) == 0.0

    def test_pearson_r_length_mismatch(self):
        with pytest.raises(ValueError):
            pearson_r([1, 2], [1, 2, 3])

    def test_covariance_matrix_matches_numpy(self):
        X = np.random.default_rng(2).normal(size=(30, 4))
        np.testing.assert_allclose(covariance_matrix(X), np.cov(X, rowvar=False))
        np.testing.assert_allclose(covariance_matrix(X, ddof=0), np.cov(X, rowvar=False, ddof=0))

    def test_covariance_matrix_validation(self):
        with pytest.raises(ValueError):
            covariance_matrix(np.arange(5))
        with pytest.raises(ValueError):
            covariance_matrix(np.ones((1, 3)))


class TestNormalization:
    def test_minmax_range(self):
        out = minmax_normalize([2, 4, 6, 8])
        np.testing.assert_allclose(out, [0, 1 / 3, 2 / 3, 1])

    def test_minmax_feature_range(self):
        out = minmax_normalize([0, 5, 10], feature_range=(-1, 1))
        np.testing.assert_allclose(out, [-1, 0, 1])
        with pytest.raises(ValueError):
            minmax_normalize([1, 2], feature_range=(1, 0))

    def test_minmax_constant_is_finite(self):
        out = minmax_normalize(np.full(4, 7.0))
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, 0.0)

    def test_minmax_per_column(self):
        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        out = minmax_normalize(X, axis=0)
        np.testing.assert_allclose(out, [[0, 0], [0.5, 0.5], [1, 1]])

    def test_zscore_properties(self):
        x = np.random.default_rng(3).normal(5, 3, size=1000)
        z = zscore_normalize(x)
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0)

    def test_zscore_constant_and_ddof(self):
        np.testing.assert_allclose(zscore_normalize([4, 4, 4]), 0.0)
        z = zscore_normalize([1, 2, 3, 4], ddof=1)
        assert z.std(ddof=1) == pytest.approx(1.0)

    def test_zscore_axis(self):
        X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        z = zscore_normalize(X, axis=1)
        np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(z.std(axis=1), 1.0)

    def test_robust_scale(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        out = robust_scale(x)
        q1, q3 = np.percentile(x, [25, 75])
        np.testing.assert_allclose(out, (x - q1) / (q3 - q1))

    def test_robust_scale_zero_iqr(self):
        out = robust_scale(np.full(5, 2.0))
        np.testing.assert_allclose(out, 0.0)

    def test_standardize_rows(self):
        X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        Z = standardize_rows(X)
        np.testing.assert_allclose(Z.mean(axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=1), 1.0)
        with pytest.raises(ValueError):
            standardize_rows(np.arange(3))


class TestWeighted:
    def test_uniform_weights_reduce_to_plain(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        assert weighted_mean(x, w) == pytest.approx(x.mean())
        assert weighted_var(x, w) == pytest.approx(x.var())
        assert weighted_std(x, w) == pytest.approx(x.std())

    def test_matches_np_average(self):
        rng = np.random.default_rng(4)
        x, w = rng.normal(size=20), rng.random(20)
        assert weighted_mean(x, w) == pytest.approx(np.average(x, weights=w))

    def test_validation(self):
        with pytest.raises(ValueError):
            weighted_mean([1, 2], [1])
        with pytest.raises(ValueError):
            weighted_mean([1, 2], [-1, 1])
        with pytest.raises(ValueError):
            weighted_mean([1, 2], [0, 0])


class TestRunningStats:
    def test_scalar_stream_matches_numpy(self):
        x = np.random.default_rng(5).normal(size=500)
        rs = RunningStats()
        for v in x:
            rs.update(v)
        assert rs.n == 500
        assert rs.mean == pytest.approx(x.mean())
        assert rs.variance() == pytest.approx(x.var())
        assert rs.variance(ddof=1) == pytest.approx(x.var(ddof=1))
        assert rs.std() == pytest.approx(x.std())
        assert rs.min == x.min()
        assert rs.max == x.max()

    def test_chunked_updates_match_numpy(self):
        x = np.random.default_rng(6).normal(size=1001)
        rs = RunningStats()
        for chunk in np.array_split(x, 7):
            rs.update(chunk)
        assert rs.mean == pytest.approx(x.mean())
        assert rs.variance() == pytest.approx(x.var())

    def test_merge_equals_single_pass(self):
        x = np.random.default_rng(7).normal(size=300)
        a = RunningStats().update(x[:100])
        b = RunningStats().update(x[100:])
        a.merge(b)
        assert a.n == 300
        assert a.mean == pytest.approx(x.mean())
        assert a.variance() == pytest.approx(x.var())

    def test_numerical_stability_with_large_offset(self):
        # Naive sum-of-squares would lose precision here.
        x = 1e9 + np.random.default_rng(8).normal(size=1000)
        rs = RunningStats().update(x)
        assert rs.variance() == pytest.approx(x.var(), rel=1e-6)

    def test_empty_state(self):
        rs = RunningStats()
        assert rs.n == 0
        assert np.isnan(rs.mean)
        assert np.isnan(rs.variance())
        rs.update([])
        assert rs.n == 0
        assert "RunningStats" in repr(rs)


class TestBootstrap:
    def test_interval_contains_true_mean(self):
        x = np.random.default_rng(9).normal(10, 2, size=200)
        low, high = bootstrap_ci(x, np.mean, n_resamples=2000, seed=0)
        assert low < 10 < high
        assert high - low < 1.5

    def test_reproducible(self):
        x = np.arange(50.0)
        assert bootstrap_ci(x, seed=1) == bootstrap_ci(x, seed=1)

    def test_median_statistic(self):
        x = np.random.default_rng(10).normal(size=100)
        low, high = bootstrap_ci(x, np.median, seed=2)
        assert low <= high

    def test_validation(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])
        with pytest.raises(ValueError):
            bootstrap_ci([1, 2, 3], confidence=1.5)


class TestRandom:
    def test_integers(self):
        out = generate_random_integers(0, 10, 100, seed=0)
        assert out.shape == (100,)
        assert out.min() >= 0
        assert out.max() < 10
        assert np.array_equal(out, generate_random_integers(0, 10, 100, seed=0))

    def test_floats_and_normal(self):
        f = generate_random_floats((3, 4), seed=0)
        assert f.shape == (3, 4)
        assert ((f >= 0) & (f < 1)).all()
        n = generate_normal_distribution(10_000, seed=0, loc=5.0, scale=0.5)
        assert n.mean() == pytest.approx(5.0, abs=0.05)
        assert n.std() == pytest.approx(0.5, abs=0.05)

    def test_generator_object_accepted(self):
        gen = np.random.default_rng(3)
        a = generate_random_floats(5, seed=gen)
        b = generate_random_floats(5, seed=gen)
        assert not np.array_equal(a, b)

    def test_random_sample(self):
        arr = np.arange(10)
        s = random_sample(arr, 5, replace=False, seed=0)
        assert len(np.unique(s)) == 5
        assert set(s).issubset(set(arr))


class TestHistogramsAndFriends:
    def test_histogram(self):
        counts, edges = histogram_binning(np.arange(10), bins=5)
        assert counts.sum() == 10
        assert len(edges) == 6

    def test_bincount(self):
        np.testing.assert_array_equal(compute_bincount([0, 1, 1, 3]), [1, 2, 0, 1])

    def test_ecdf(self):
        x, y = ecdf([3, 1, 2])
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1])
        with pytest.raises(ValueError):
            ecdf([])

    def test_entropy(self):
        assert entropy([1, 1], base=2) == pytest.approx(1.0)
        assert entropy([1, 0, 0]) == 0.0
        assert entropy([0.25, 0.25, 0.25, 0.25], base=2) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            entropy([-1, 1])
        with pytest.raises(ValueError):
            entropy([0, 0])

    def test_moving_average_valid(self):
        out = moving_average([1, 2, 3, 4, 5], window=3)
        np.testing.assert_allclose(out, [2, 3, 4])

    def test_moving_average_same(self):
        out = moving_average([1, 2, 3, 4, 5], window=2, mode="same")
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1:], [1.5, 2.5, 3.5, 4.5])

    def test_moving_average_matches_convolve(self):
        x = np.random.default_rng(11).normal(size=100)
        np.testing.assert_allclose(
            moving_average(x, 7), np.convolve(x, np.ones(7) / 7, mode="valid")
        )

    def test_moving_average_validation(self):
        with pytest.raises(ValueError):
            moving_average([1, 2, 3], window=0)
        with pytest.raises(ValueError):
            moving_average([1, 2, 3], window=4)
        with pytest.raises(ValueError):
            moving_average([1, 2, 3], window=2, mode="bogus")  # type: ignore[arg-type]
