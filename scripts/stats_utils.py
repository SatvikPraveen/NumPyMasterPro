"""
stats_utils.py

Statistical and random-sampling utilities built on NumPy.

Highlights
----------
- Axis-aware, division-safe normalisers (constant slices map to zero instead
  of NaN/inf) with optional NaN handling
- Descriptive summary including skewness and excess kurtosis
- ``RunningStats``: Welford's online algorithm with Chan's parallel merge, so
  streams and chunks can be accumulated without a second pass
- Weighted mean/variance, covariance matrix from first principles
- Vectorised bootstrap confidence intervals
- Moving averages via ``sliding_window_view`` (no Python loops)
- Empirical CDF and Shannon entropy helpers
- Reproducible sampling through ``numpy.random.Generator``

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

FloatArray = npt.NDArray[np.floating]
RandomState = int | np.random.Generator | None
NanPolicy = Literal["propagate", "omit", "raise"]

__all__ = [
    "RunningStats",
    "bootstrap_ci",
    "compute_bincount",
    "compute_correlation",
    "compute_covariance",
    "covariance_matrix",
    "ecdf",
    "entropy",
    "generate_normal_distribution",
    "generate_random_floats",
    "generate_random_integers",
    "histogram_binning",
    "minmax_normalize",
    "moving_average",
    "pearson_r",
    "quantiles",
    "random_sample",
    "robust_scale",
    "standardize_rows",
    "summarize_array",
    "weighted_mean",
    "weighted_std",
    "weighted_var",
    "zscore_normalize",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_rng(seed: RandomState = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _apply_nan_policy(arr: FloatArray, nan_policy: NanPolicy) -> FloatArray:
    """Return ``arr`` filtered according to ``nan_policy`` (for 1-D reductions)."""
    has_nan = bool(np.isnan(arr).any())
    if nan_policy == "raise" and has_nan:
        raise ValueError("Input contains NaN values")
    if nan_policy == "omit" and has_nan:
        return arr[~np.isnan(arr)]
    return arr


def _safe_divide(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Element-wise division that yields 0 wherever the denominator is 0."""
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


# ---------------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------------
def summarize_array(arr: npt.ArrayLike, nan_policy: NanPolicy = "propagate") -> dict[str, float]:
    """
    Core descriptive statistics of a (flattened) array.

    Returns mean, median, std, var, min, max, range, quartiles, IQR, plus
    ``skewness`` (Fisher) and ``kurtosis`` (excess, so a normal distribution
    scores 0). ``count`` is the number of values actually used.

    Parameters
    ----------
    arr : array_like
    nan_policy : {"propagate", "omit", "raise"}
        How to treat NaN: propagate into the results, drop them, or raise.
    """
    x = _apply_nan_policy(np.asarray(arr, dtype=float).ravel(), nan_policy)
    if x.size == 0:
        raise ValueError("Cannot summarise an empty array")

    mean = float(np.mean(x))
    std = float(np.std(x))
    q1, q3 = np.percentile(x, [25, 75])
    centred = x - mean
    m2 = float(np.mean(centred**2))
    if m2 > 0:
        skew = float(np.mean(centred**3) / m2**1.5)
        kurt = float(np.mean(centred**4) / m2**2 - 3.0)
    else:
        skew = kurt = 0.0

    return {
        "count": int(x.size),
        "mean": mean,
        "median": float(np.median(x)),
        "std": std,
        "var": float(np.var(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.ptp(x)),
        "25th_percentile": float(q1),
        "75th_percentile": float(q3),
        "IQR": float(q3 - q1),
        "skewness": skew,
        "kurtosis": kurt,
    }


def quantiles(arr: npt.ArrayLike, q: tuple[float, ...] = (0.25, 0.5, 0.75)) -> FloatArray:
    """Quantiles of ``arr`` (default: Q1, median, Q3)."""
    return np.quantile(np.asarray(arr, dtype=float), q)


# ---------------------------------------------------------------------------
# 2. Correlation & covariance
# ---------------------------------------------------------------------------
def compute_covariance(x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """2x2 sample covariance matrix of two 1-D arrays (``np.cov``)."""
    return np.cov(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def compute_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """2x2 Pearson correlation matrix of two 1-D arrays (``np.corrcoef``)."""
    return np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def pearson_r(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """
    Pearson correlation coefficient computed from first principles.

    Returns 0.0 if either input has zero variance (instead of NaN).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    return float((xc * yc).sum() / denom) if denom > 0 else 0.0


def covariance_matrix(X: npt.ArrayLike, ddof: int = 1) -> FloatArray:
    """
    Covariance matrix of the columns of ``X`` (rows are observations).

    Implemented as ``Xcᵀ Xc / (n - ddof)`` with ``Xc`` the column-centred data,
    which is what ``np.cov(X, rowvar=False)`` does under the hood.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_observations, n_features)")
    n = X.shape[0]
    if n - ddof <= 0:
        raise ValueError(f"Need more than {ddof} observations for ddof={ddof}")
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (n - ddof)


# ---------------------------------------------------------------------------
# 3. Normalisation
# ---------------------------------------------------------------------------
def minmax_normalize(
    arr: npt.ArrayLike,
    axis: int | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> FloatArray:
    """
    Scale values to ``feature_range`` using min/max along ``axis``.

    Constant slices (max == min) map to the lower bound instead of NaN.

    Parameters
    ----------
    arr : array_like
    axis : int or None
        ``None`` normalises over the whole array; an int normalises per slice
        (e.g. ``axis=0`` scales each column of a 2-D array independently).
    feature_range : (low, high)
    """
    x = np.asarray(arr, dtype=float)
    lo, hi = feature_range
    if hi <= lo:
        raise ValueError("feature_range must satisfy low < high")
    mn = np.min(x, axis=axis, keepdims=True)
    mx = np.max(x, axis=axis, keepdims=True)
    scaled = _safe_divide(x - mn, mx - mn)
    return lo + scaled * (hi - lo)


def zscore_normalize(arr: npt.ArrayLike, axis: int | None = None, ddof: int = 0) -> FloatArray:
    """
    Standardise to mean 0 / std 1 along ``axis``.

    Slices with zero variance are returned as all-zeros instead of NaN.
    """
    x = np.asarray(arr, dtype=float)
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, ddof=ddof, keepdims=True)
    # A slice whose range is exactly zero is constant even if rounding in the
    # mean leaves a std of a few ulps; force those denominators to zero.
    std = np.where(np.ptp(x, axis=axis, keepdims=True) == 0, 0.0, std)
    return _safe_divide(x - mean, std)


def robust_scale(arr: npt.ArrayLike, axis: int | None = None) -> FloatArray:
    """
    Scale by the interquartile range: ``(x - Q1) / (Q3 - Q1)``.

    Robust to outliers; a zero IQR yields ``x - Q1``.
    """
    x = np.asarray(arr, dtype=float)
    q1 = np.percentile(x, 25, axis=axis, keepdims=True)
    q3 = np.percentile(x, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    shifted = x - q1
    return np.where(iqr != 0, _safe_divide(shifted, iqr), shifted)


def standardize_rows(arr: npt.ArrayLike) -> FloatArray:
    """Standardise each row of a 2-D array to mean 0, std 1 (``zscore_normalize`` with ``axis=1``)."""
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError("standardize_rows expects a 2-D array")
    return zscore_normalize(x, axis=1)


# ---------------------------------------------------------------------------
# 4. Weighted statistics
# ---------------------------------------------------------------------------
def weighted_mean(values: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """Weighted arithmetic mean ``Σ wᵢxᵢ / Σ wᵢ``."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must not sum to zero")
    return float((v * w).sum() / total)


def weighted_var(values: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """Weighted (population) variance ``Σ wᵢ(xᵢ - μ_w)² / Σ wᵢ``."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = weighted_mean(v, w)
    return float((w * (v - mu) ** 2).sum() / w.sum())


def weighted_std(values: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """Weighted (population) standard deviation."""
    return float(np.sqrt(weighted_var(values, weights)))


# ---------------------------------------------------------------------------
# 5. Online statistics (Welford / Chan)
# ---------------------------------------------------------------------------
class RunningStats:
    """
    Numerically stable single-pass mean/variance accumulator.

    Uses Welford's update for individual observations and Chan et al.'s
    parallel formula for merging batches, so it can consume a stream of
    scalars, chunks of an array too large to hold in memory, or partial
    results from separate workers.

    Examples
    --------
    >>> rs = RunningStats()
    >>> for chunk in np.array_split(np.arange(10.0), 3):
    ...     rs.update(chunk)
    >>> rs.n, rs.mean, round(rs.variance(), 4)
    (10, 4.5, 8.25)
    """

    __slots__ = ("_m2", "_mean", "_n", "max", "min")

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = np.inf
        self.max = -np.inf

    # -- accumulation ------------------------------------------------------
    def update(self, x: npt.ArrayLike) -> RunningStats:
        """Absorb a scalar or an array of observations."""
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 0:
            return self
        if arr.size == 1:
            self._push_scalar(float(arr[0]))
        else:
            self._merge_moments(arr.size, float(arr.mean()), float(((arr - arr.mean()) ** 2).sum()))
            self.min = min(self.min, float(arr.min()))
            self.max = max(self.max, float(arr.max()))
        return self

    def merge(self, other: RunningStats) -> RunningStats:
        """Fold another accumulator into this one (Chan's parallel merge)."""
        if other._n:
            self._merge_moments(other._n, other._mean, other._m2)
            self.min = min(self.min, other.min)
            self.max = max(self.max, other.max)
        return self

    def _push_scalar(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def _merge_moments(self, n_b: int, mean_b: float, m2_b: float) -> None:
        n_a = self._n
        n = n_a + n_b
        delta = mean_b - self._mean
        self._mean += delta * n_b / n
        self._m2 += m2_b + delta**2 * n_a * n_b / n
        self._n = n

    # -- results -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n else float("nan")

    def variance(self, ddof: int = 0) -> float:
        if self._n - ddof <= 0:
            return float("nan")
        return self._m2 / (self._n - ddof)

    def std(self, ddof: int = 0) -> float:
        return float(np.sqrt(self.variance(ddof)))

    def __repr__(self) -> str:
        return f"RunningStats(n={self._n}, mean={self.mean:.6g}, std={self.std():.6g})"


# ---------------------------------------------------------------------------
# 6. Resampling
# ---------------------------------------------------------------------------
def bootstrap_ci(
    arr: npt.ArrayLike,
    statistic: Callable[..., npt.ArrayLike] = np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: RandomState = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for a statistic.

    All resamples are drawn in one ``(n_resamples, n)`` index array and the
    statistic is evaluated once along ``axis=1``, so no Python-level loop.

    Parameters
    ----------
    arr : array_like, 1-D
    statistic : callable accepting ``axis``
        e.g. ``np.mean``, ``np.median``, ``np.std``.
    n_resamples : int
    confidence : float in (0, 1)
    seed : int | Generator | None

    Returns
    -------
    (low, high) : tuple of float
    """
    x = np.asarray(arr, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    rng = _as_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    stats = np.asarray(statistic(x[idx], axis=1), dtype=float)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# 7. Random sampling (Generator-based, reproducible)
# ---------------------------------------------------------------------------
def generate_random_integers(
    low: int, high: int, size: int | tuple[int, ...], seed: RandomState = None
) -> npt.NDArray[np.int64]:
    """Random integers in ``[low, high)``."""
    return _as_rng(seed).integers(low, high, size=size)


def generate_random_floats(size: int | tuple[int, ...], seed: RandomState = None) -> FloatArray:
    """Uniform random floats in ``[0, 1)``."""
    return _as_rng(seed).random(size)


def generate_normal_distribution(
    size: int | tuple[int, ...], seed: RandomState = None, loc: float = 0.0, scale: float = 1.0
) -> FloatArray:
    """Normally distributed values with mean ``loc`` and std ``scale``."""
    return _as_rng(seed).normal(loc=loc, scale=scale, size=size)


def random_sample(
    arr: npt.ArrayLike, sample_size: int, replace: bool = True, seed: RandomState = None
) -> np.ndarray:
    """Sample ``sample_size`` elements from a 1-D array."""
    return _as_rng(seed).choice(np.asarray(arr), size=sample_size, replace=replace)


# ---------------------------------------------------------------------------
# 8. Histograms, ECDF, entropy, moving averages
# ---------------------------------------------------------------------------
def histogram_binning(arr: npt.ArrayLike, bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Histogram counts and bin edges."""
    return np.histogram(np.asarray(arr), bins=bins)


def compute_bincount(arr: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """Count of each non-negative integer value."""
    return np.bincount(np.asarray(arr))


def ecdf(arr: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """
    Empirical cumulative distribution function.

    Returns
    -------
    (x, y) : sorted values and the fraction of observations ``<= x``.
    """
    x = np.sort(np.asarray(arr, dtype=float).ravel())
    if x.size == 0:
        raise ValueError("ecdf of an empty array is undefined")
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def entropy(counts: npt.ArrayLike, base: float | None = None) -> float:
    """
    Shannon entropy of a discrete distribution given counts or probabilities.

    Zero entries are ignored (``0·log 0 := 0``). Natural log by default;
    pass ``base=2`` for bits.
    """
    p = np.asarray(counts, dtype=float).ravel()
    if np.any(p < 0):
        raise ValueError("counts must be non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("counts must not sum to zero")
    p = p / total
    p = p[p > 0]  # filter *after* normalising: tiny counts can underflow to 0
    h = float(-(p * np.log(p)).sum())
    return h / np.log(base) if base is not None else h


def moving_average(
    arr: npt.ArrayLike, window: int, mode: Literal["valid", "same"] = "valid"
) -> FloatArray:
    """
    Simple moving average using a strided view (no copies, no loops).

    Parameters
    ----------
    arr : array_like, 1-D
    window : int, ``1 <= window <= len(arr)``
    mode : {"valid", "same"}
        ``"valid"`` returns ``len(arr) - window + 1`` values; ``"same"`` pads
        the front with NaN so the output aligns with the input.
    """
    x = np.asarray(arr, dtype=float).ravel()
    if not 1 <= window <= x.size:
        raise ValueError(f"window must be in [1, {x.size}]")
    out = sliding_window_view(x, window).mean(axis=1)
    if mode == "same":
        return np.concatenate([np.full(window - 1, np.nan), out])
    if mode == "valid":
        return out
    raise ValueError("mode must be 'valid' or 'same'")
