"""
kmeans_utils.py

K-Means clustering implemented from scratch with NumPy only.

Highlights
----------
- ``k-means++`` seeding (D² sampling) in addition to uniform random seeding
- Multiple restarts (``n_init``) keeping the lowest-inertia solution
- Memory-efficient pairwise squared distances via the
  ``||x||² - 2·x·c + ||c||²`` expansion (O(n·k) instead of O(n·k·d) temporaries)
- Vectorised centroid updates with ``np.add.at`` and ``np.bincount``
- Empty-cluster repair (re-seed from the point farthest from its centroid)
- Reproducibility through ``numpy.random.Generator`` (no global state)
- Silhouette score and an automatic elbow detector for model selection
- A small scikit-learn-style ``KMeans`` class on top of the functional API

The functional API (``generate_data``, ``initialize_centroids``,
``assign_clusters``, ``update_centroids``, ``kmeans``, ``compute_inertia``,
``compute_cluster_inertia``) is backward compatible with earlier versions.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
InitMethod = Literal["k-means++", "random"]
RandomState = int | np.random.Generator | None

__all__ = [
    "KMeans",
    "KMeansResult",
    "assign_clusters",
    "compute_cluster_inertia",
    "compute_inertia",
    "elbow_point",
    "generate_data",
    "initialize_centroids",
    "kmeans",
    "pairwise_sq_distances",
    "run_kmeans",
    "silhouette_score",
    "update_centroids",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_rng(seed: RandomState = None) -> np.random.Generator:
    """Coerce ``None`` / int / Generator into a ``numpy.random.Generator``."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_2d(X: npt.ArrayLike, name: str = "X") -> FloatArray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def pairwise_sq_distances(X: npt.ArrayLike, C: npt.ArrayLike) -> FloatArray:
    """
    Squared Euclidean distances between every row of ``X`` and every row of ``C``.

    Uses the identity ``||x - c||² = ||x||² - 2·x·c + ||c||²`` so that only an
    ``(n, k)`` matrix is materialised, instead of the ``(n, k, d)`` temporary
    created by ``X[:, None, :] - C[None, :, :]``.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    C : array_like, shape (n_centroids, n_features)

    Returns
    -------
    ndarray, shape (n_samples, n_centroids)
        Squared distances, clipped at zero to absorb floating-point round-off.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    c_sq = np.einsum("ij,ij->i", C, C)[None, :]
    d2 = x_sq - 2.0 * (X @ C.T) + c_sq
    np.maximum(d2, 0.0, out=d2)
    return d2


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def generate_data(
    n_per_cluster: int = 50,
    centers: Sequence[Sequence[float]] = ((2.0, 2.0), (7.0, 7.0), (2.0, 7.0)),
    std: float = 1.0,
    seed: RandomState = 42,
) -> FloatArray:
    """
    Generate isotropic Gaussian blobs for clustering demos.

    Defaults reproduce the classic 3-cluster, 150-point dataset used in the
    notebooks: 50 points each around (2, 2), (7, 7) and (2, 7).

    Parameters
    ----------
    n_per_cluster : int
        Number of samples drawn around each centre.
    centers : sequence of coordinate sequences
        Cluster centres; ``len(centers)`` clusters of dimension ``len(centers[0])``.
    std : float
        Standard deviation of each blob.
    seed : int | Generator | None
        Seed or Generator for reproducibility.

    Returns
    -------
    ndarray, shape (n_per_cluster * len(centers), n_features)
    """
    rng = _as_rng(seed)
    centers_arr = np.asarray(centers, dtype=float)
    blobs = [
        rng.normal(loc=c, scale=std, size=(n_per_cluster, centers_arr.shape[1]))
        for c in centers_arr
    ]
    return np.vstack(blobs)


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------
def initialize_centroids(
    X: npt.ArrayLike,
    k: int,
    method: InitMethod = "random",
    rng: RandomState = None,
) -> FloatArray:
    """
    Choose ``k`` initial centroids from the rows of ``X``.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    k : int
        Number of clusters, ``1 <= k <= n_samples``.
    method : {"random", "k-means++"}
        ``"random"`` picks ``k`` distinct rows uniformly.
        ``"k-means++"`` picks the first centre uniformly and each subsequent
        centre with probability proportional to its squared distance to the
        nearest already-chosen centre (Arthur & Vassilvitskii, 2007).
    rng : int | Generator | None
        Seed or Generator.

    Returns
    -------
    ndarray, shape (k, n_features)

    Raises
    ------
    ValueError
        If ``k`` is out of range or ``method`` is unknown.
    """
    X = _validate_2d(X)
    n_samples = X.shape[0]
    if not 1 <= k <= n_samples:
        raise ValueError(
            f"❌ Cannot initialize {k} centroids with {n_samples} samples. Please choose 1 ≤ k ≤ {n_samples}."
        )
    rng = _as_rng(rng)

    if method == "random":
        return X[rng.choice(n_samples, size=k, replace=False)].copy()

    if method == "k-means++":
        centroids = np.empty((k, X.shape[1]), dtype=float)
        centroids[0] = X[rng.integers(n_samples)]
        closest_d2 = pairwise_sq_distances(X, centroids[:1])[:, 0]
        for i in range(1, k):
            total = closest_d2.sum()
            if total <= 0.0:  # all remaining points coincide with a centre
                idx = int(rng.integers(n_samples))
            else:
                idx = int(rng.choice(n_samples, p=closest_d2 / total))
            centroids[i] = X[idx]
            new_d2 = pairwise_sq_distances(X, centroids[i : i + 1])[:, 0]
            np.minimum(closest_d2, new_d2, out=closest_d2)
        return centroids

    raise ValueError(f"Unknown init method {method!r}; expected 'random' or 'k-means++'.")


def assign_clusters(X: npt.ArrayLike, centroids: npt.ArrayLike) -> IntArray:
    """
    Assign each sample to the nearest centroid (squared Euclidean distance).

    Returns
    -------
    ndarray of int, shape (n_samples,)
    """
    return np.argmin(pairwise_sq_distances(X, centroids), axis=1)


def update_centroids(
    X: npt.ArrayLike,
    labels: npt.ArrayLike,
    k: int,
    previous: npt.ArrayLike | None = None,
) -> FloatArray:
    """
    Recompute centroids as the mean of their assigned samples.

    The update is fully vectorised: per-cluster sums come from ``np.add.at``
    and counts from ``np.bincount``. Empty clusters are repaired rather than
    producing NaN:

    * if ``previous`` centroids are given, an empty cluster is re-seeded at the
      sample farthest from its current centroid (the standard k-means fix);
    * otherwise it is re-seeded at the sample farthest from the non-empty
      centroids.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    labels : array_like of int, shape (n_samples,)
    k : int
    previous : array_like, shape (k, n_features), optional
        Centroids from the previous iteration, used for empty-cluster repair.

    Returns
    -------
    ndarray, shape (k, n_features)
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    sums = np.zeros((k, X.shape[1]), dtype=float)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=k).astype(float)

    non_empty = counts > 0
    centroids = np.empty_like(sums)
    centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

    empty_idx = np.flatnonzero(~non_empty)
    if empty_idx.size:
        if previous is not None:
            centroids[~non_empty] = np.asarray(previous, dtype=float)[~non_empty]
        # Re-seed each empty cluster at the point currently farthest from its centroid.
        reference = centroids if previous is not None else centroids[non_empty]
        d2 = pairwise_sq_distances(X, reference).min(axis=1)
        for idx in empty_idx:
            far = int(np.argmax(d2))
            centroids[idx] = X[far]
            d2[far] = 0.0  # don't pick the same point twice
    return centroids


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KMeansResult:
    """Outcome of a single k-means fit."""

    centroids: FloatArray
    labels: IntArray
    inertia: float
    n_iter: int
    converged: bool
    inertia_history: list[float] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])


def _single_run(
    X: FloatArray,
    k: int,
    max_iters: int,
    tol: float,
    init: InitMethod,
    rng: np.random.Generator,
) -> KMeansResult:
    centroids = initialize_centroids(X, k, method=init, rng=rng)
    history: list[float] = []
    converged = False
    n_iter = 0

    for iteration in range(1, max_iters + 1):
        n_iter = iteration
        d2 = pairwise_sq_distances(X, centroids)
        labels = np.argmin(d2, axis=1)
        history.append(float(d2[np.arange(X.shape[0]), labels].sum()))

        new_centroids = update_centroids(X, labels, k, previous=centroids)
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            converged = True
            break

    # Final assignment so labels/inertia are consistent with the returned centroids.
    d2 = pairwise_sq_distances(X, centroids)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(X.shape[0]), labels].sum())
    return KMeansResult(centroids, labels, inertia, n_iter, converged, history)


def run_kmeans(
    X: npt.ArrayLike,
    k: int = 3,
    max_iters: int = 100,
    tol: float = 1e-4,
    init: InitMethod = "k-means++",
    n_init: int = 1,
    seed: RandomState = None,
) -> KMeansResult:
    """
    Lloyd's algorithm with optional restarts. Returns the lowest-inertia run.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    k : int
        Number of clusters.
    max_iters : int
        Maximum Lloyd iterations per run.
    tol : float
        Convergence threshold on the Frobenius norm of the centroid shift.
    init : {"k-means++", "random"}
        Seeding strategy.
    n_init : int
        Number of independent runs; the best (lowest inertia) is returned.
    seed : int | Generator | None
        Seed or Generator for reproducibility.

    Returns
    -------
    KMeansResult
    """
    X = _validate_2d(X)
    if n_init < 1:
        raise ValueError("n_init must be >= 1")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    rng = _as_rng(seed)

    best: KMeansResult | None = None
    for _ in range(n_init):
        result = _single_run(X, k, max_iters, tol, init, rng)
        if best is None or result.inertia < best.inertia:
            best = result
    assert best is not None  # for type checkers; n_init >= 1 guarantees a result
    return best


def kmeans(
    X: npt.ArrayLike,
    k: int = 3,
    max_iters: int = 100,
    tol: float = 1e-4,
    init: InitMethod = "k-means++",
    n_init: int = 1,
    seed: RandomState = None,
) -> tuple[FloatArray, IntArray]:
    """
    Convenience wrapper around :func:`run_kmeans` returning ``(centroids, labels)``.

    Kept for backward compatibility with the notebooks and the Streamlit app.
    """
    result = run_kmeans(X, k=k, max_iters=max_iters, tol=tol, init=init, n_init=n_init, seed=seed)
    return result.centroids, result.labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_cluster_inertia(
    X: npt.ArrayLike, centroids: npt.ArrayLike, labels: npt.ArrayLike
) -> float:
    """
    Within-cluster sum of squared distances (inertia / WCSS).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
    centroids : array_like, shape (k, n_features)
    labels : array_like of int, shape (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    labels = np.asarray(labels)
    return float(np.sum((X - centroids[labels]) ** 2))


def compute_inertia(
    X: npt.ArrayLike,
    k_range: Iterable[int],
    n_init: int = 3,
    seed: RandomState = None,
    **kwargs: object,
) -> list[float]:
    """
    Inertia for each ``k`` in ``k_range`` (input for the Elbow Method).

    Each ``k`` is fitted with ``n_init`` restarts so the curve is monotone in
    practice rather than jagged from unlucky seeds. Extra keyword arguments are
    forwarded to :func:`run_kmeans`.
    """
    rng = _as_rng(seed)
    return [run_kmeans(X, k=k, n_init=n_init, seed=rng, **kwargs).inertia for k in k_range]  # type: ignore[arg-type]


def elbow_point(k_values: Sequence[int], inertias: Sequence[float]) -> int:
    """
    Pick the "elbow" of an inertia curve.

    Uses the maximum-distance-to-chord heuristic (a simplified Kneedle):
    normalise both axes to [0, 1], draw the chord from the first to the last
    point, and return the ``k`` whose point lies farthest below that chord.

    Parameters
    ----------
    k_values : sequence of int
    inertias : sequence of float, same length as ``k_values``

    Returns
    -------
    int
        The ``k`` at the elbow. With fewer than three points the first ``k``
        is returned.
    """
    ks = np.asarray(k_values, dtype=float)
    ys = np.asarray(inertias, dtype=float)
    if ks.shape != ys.shape:
        raise ValueError("k_values and inertias must have the same length")
    if ks.size < 3:
        return int(k_values[0])

    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y_span = ys[0] - ys[-1]
    y = (ys - ys[-1]) / y_span if y_span != 0 else np.zeros_like(ys)
    # Chord from (0, 1) to (1, 0): distance below the line y = 1 - x.
    gap = (1.0 - x) - y
    return int(k_values[int(np.argmax(gap))])


def silhouette_score(X: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """
    Mean silhouette coefficient over all samples, computed with NumPy only.

    For sample *i* with intra-cluster mean distance *a(i)* and the smallest
    mean distance to any other cluster *b(i)*::

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Singleton clusters contribute ``s = 0``. This materialises the full
    ``(n, n)`` distance matrix, so it is intended for small/medium datasets.

    Returns
    -------
    float in [-1, 1]

    Raises
    ------
    ValueError
        If fewer than two distinct clusters are present.
    """
    X = _validate_2d(X)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("silhouette_score requires at least 2 distinct clusters")
    if unique.size >= X.shape[0]:
        raise ValueError("silhouette_score requires fewer clusters than samples")

    D = np.sqrt(pairwise_sq_distances(X, X))
    onehot = (labels[:, None] == unique[None, :]).astype(float)  # (n, k)
    counts = onehot.sum(axis=0)  # (k,)
    # Sum of distances from each point to each cluster: (n, k)
    dist_sums = D @ onehot
    own = onehot.astype(bool)

    own_counts = counts[np.argmax(own, axis=1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = dist_sums[own] / (own_counts - 1)
        mean_other = dist_sums / counts[None, :]
        mean_other[own] = np.inf
        b = mean_other.min(axis=1)
        s = np.where(own_counts > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(np.mean(s))


# ---------------------------------------------------------------------------
# Estimator-style wrapper
# ---------------------------------------------------------------------------
class KMeans:
    """
    Minimal scikit-learn-style estimator around :func:`run_kmeans`.

    Examples
    --------
    >>> X = generate_data(seed=0)
    >>> km = KMeans(n_clusters=3, seed=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 2)
    >>> km.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        init: InitMethod = "k-means++",
        n_init: int = 10,
        max_iters: int = 300,
        tol: float = 1e-4,
        seed: RandomState = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.result_: KMeansResult | None = None

    # -- fitted attributes -------------------------------------------------
    def _check_fitted(self) -> KMeansResult:
        if self.result_ is None:
            raise RuntimeError("This KMeans instance is not fitted yet. Call fit() first.")
        return self.result_

    @property
    def cluster_centers_(self) -> FloatArray:
        return self._check_fitted().centroids

    @property
    def labels_(self) -> IntArray:
        return self._check_fitted().labels

    @property
    def inertia_(self) -> float:
        return self._check_fitted().inertia

    @property
    def n_iter_(self) -> int:
        return self._check_fitted().n_iter

    # -- API ---------------------------------------------------------------
    def fit(self, X: npt.ArrayLike) -> KMeans:
        self.result_ = run_kmeans(
            X,
            k=self.n_clusters,
            max_iters=self.max_iters,
            tol=self.tol,
            init=self.init,
            n_init=self.n_init,
            seed=self.seed,
        )
        return self

    def predict(self, X: npt.ArrayLike) -> IntArray:
        return assign_clusters(_validate_2d(X), self.cluster_centers_)

    def fit_predict(self, X: npt.ArrayLike) -> IntArray:
        return self.fit(X).labels_

    def transform(self, X: npt.ArrayLike) -> FloatArray:
        """Euclidean distance from each sample to every centroid, shape (n, k)."""
        return np.sqrt(pairwise_sq_distances(_validate_2d(X), self.cluster_centers_))

    def score(self, X: npt.ArrayLike) -> float:
        """Negative inertia on ``X`` (higher is better), mirroring scikit-learn."""
        X = _validate_2d(X)
        return -compute_cluster_inertia(X, self.cluster_centers_, self.predict(X))

    def __repr__(self) -> str:
        return (
            f"KMeans(n_clusters={self.n_clusters}, init={self.init!r}, n_init={self.n_init}, "
            f"max_iters={self.max_iters}, tol={self.tol}, seed={self.seed!r})"
        )
