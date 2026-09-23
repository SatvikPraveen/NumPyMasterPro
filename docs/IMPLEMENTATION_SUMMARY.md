# 🏗️ Implementation Notes — NumPyMasterPro 2.0

Design decisions behind the `scripts/` package, for readers who want to know
*why* the code looks the way it does. For the user-facing overview see the
[README](../README.md); for the test suite see [TESTING.md](TESTING.md).

---

## Guiding principles

1. **NumPy only for the numerics.** Every algorithm (k-means++, PCA, ridge,
   Welford, Gram-Schmidt, power iteration) is written against `numpy` so the
   mechanics are visible. pandas appears only for tabular display, and is
   imported lazily.
2. **Reproducible by construction.** All randomness goes through
   `numpy.random.Generator`. Public functions take `seed: int | Generator | None`
   and coerce it with a tiny `_as_rng` helper; nothing touches
   `np.random.seed`. Passing a `Generator` lets callers thread one stream
   through several calls (e.g. `compute_inertia` reuses one generator across
   every `k`).
3. **Fail loudly, never silently produce NaN.** Inputs are validated
   (`_validate_2d`, `_as_square`, `_as_matrix`) and edge cases that NumPy would
   turn into `nan`/`inf` (empty clusters, constant slices, zero weights,
   singular triangular systems) raise `ValueError` or return a documented
   safe value.
4. **Backward compatible with the notebooks.** The notebooks call the
   utilities by name with positional arguments; those names and positions are
   frozen. New behaviour is added through keyword arguments with defaults.
5. **Typed and linted.** `from __future__ import annotations`, PEP 604 unions,
   `numpy.typing` aliases, `mypy` with `no_implicit_optional`, and ruff's
   `NPY` rule set (which forbids the legacy random API).

---

## Module by module

### `kmeans_utils.py`

| Concern | Decision |
| --- | --- |
| Distances | `pairwise_sq_distances` uses ‖x‖² − 2x·c + ‖c‖² via `einsum` + one matmul. The naive `X[:, None] - C[None]` allocates an `(n, k, d)` temporary; this allocates `(n, k)`. Values are clipped at 0 to absorb cancellation error. |
| Seeding | `k-means++` draws each new centre with probability ∝ D² to the nearest chosen centre. A degenerate case (all remaining D² are 0) falls back to a uniform draw. |
| Update | Cluster sums via `np.add.at`, counts via `np.bincount(minlength=k)`. Empty clusters are re-seeded at the point farthest from the current centroids, one at a time, so two empty clusters never collapse onto the same point. |
| Convergence | Frobenius norm of the centroid shift ≤ `tol`. After the loop, labels and inertia are recomputed against the final centroids so the returned triple is self-consistent even when `max_iters` is hit. |
| Restarts | `run_kmeans(n_init=…)` keeps the lowest-inertia run; `inertia_history` is the per-iteration WCSS of that run, which the app plots. |
| Model selection | `silhouette_score` builds the full `(n, n)` distance matrix and computes a/b with one matmul against a one-hot label matrix (no Python loops; intended for small/medium n). `elbow_point` normalises both axes and picks the point farthest below the first→last chord. |
| Estimator | `KMeans` is a thin façade with `fit/predict/transform/score`, `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, mirroring scikit-learn names so it can be swapped in for comparison. |

### `linear_algebra_utils.py`

* `closed_form_linear_regression` defaults to `np.linalg.lstsq` (SVD). The
  normal-equation path `inv(XᵀX)Xᵀy` squares the condition number and is kept
  only as `method="normal"` for demonstration; `"qr"` and `"cholesky"` are
  provided so the three can be compared in the notebooks.
* `ridge_regression` solves `(XᵀX + αI)w = Xᵀy` with Cholesky and zeroes the
  penalty on the intercept row, matching scikit-learn.
* `forward_substitution` / `back_substitution` are written row by row (each
  row is one dot product) so the algorithm is legible; they read only the
  relevant triangle. `cholesky_solve` and `qr_solve` are built on them.
* `gram_schmidt` is the *modified* variant: each remaining column is
  orthogonalised against `q_j` immediately (`V[:, j+1:] -= outer(q_j, r_j)`),
  which is far more stable than classical GS.
* `pca` centres, takes the thin SVD, and normalises component signs so the
  largest-magnitude entry is positive, making output deterministic across
  LAPACK builds.

### `stats_utils.py`

* Normalisers call `_safe_divide`, which writes 0 where the denominator is 0.
  `zscore_normalize` additionally treats zero-range slices as constant, because
  identical values can still yield a std of a few ulps after mean rounding.
* `RunningStats` keeps `(n, mean, M2)`; scalars use Welford's update and
  batches/merges use Chan et al.'s pairwise formula, so chunked and streamed
  data give identical results to a single NumPy pass (verified by property
  tests, including data with a 1e9 offset).
* `bootstrap_ci` draws one `(n_resamples, n)` index matrix and evaluates the
  statistic along `axis=1`, so the caller's statistic must accept `axis`.
* `entropy` normalises *before* filtering zeros; filtering first allowed tiny
  counts to underflow to exactly 0 and produce `0·log 0 = nan`.
* `moving_average` is a `sliding_window_view(...).mean(axis=1)`, no loop.

### `array_utils.py` and `perf_utils.py`

* `array_flags` reads `ndarray.flags` attributes instead of parsing the
  string form, which changed between NumPy releases.
* `broadcast_result_shape` implements the broadcasting rule by hand
  (right-align, dims equal or 1) and is property-tested against
  `np.broadcast_shapes`; `explain_broadcast` renders the same walk as text.
* `rolling_windows` and `strided_blocks` return read-only views; the latter
  computes strides explicitly with `as_strided`, which is the classic way to
  do block-wise image operations without copies.
* `timeit_compare` reports the minimum over repeats (least noisy) and asserts
  all implementations produce `allclose` outputs, so a benchmark can never
  quietly crown a wrong-but-fast version.

### `io_utils.py`

* Every function accepts `str | os.PathLike`, creates parent directories, and
  returns the path actually written (NumPy appends `.npy`/`.npz` when missing).
* `np.load` is always called with `allow_pickle=False`.
* `load_npz(as_dict=True)` reads eagerly inside a context manager so the file
  handle is closed; the default returns the lazy `NpzFile` for large archives.

---

## Tooling

| Tool | Role |
| --- | --- |
| `pyproject.toml` | Single source of metadata and configuration for ruff, mypy, pytest and coverage. `pytest.ini`'s `[coverage:*]` sections were never read by pytest; they now live under `[tool.coverage]`. |
| ruff | Lint + import sorting + formatting (replaces flake8, isort, black). Rule sets: `E W F I UP B NPY RUF SIM PT`. |
| mypy | `no_implicit_optional`, `check_untyped_defs`, `warn_unused_ignores`; runs against `scripts/`. |
| pytest + hypothesis | Unit, property-based and headless-app layers; branch coverage ≈ 96%. |
| pre-commit | ruff, ruff-format, mypy and hygiene hooks. |
| GitHub Actions | lint → tests (3.10–3.13 × 3 OSes) → notebooks → bandit/pip-audit → Docker build + smoke test → status gate. No step masks failures. |
| Docker | Multi-stage `python:3.12-slim` image; dependencies resolved with uv into `/opt/venv`; unprivileged user; healthcheck; compose profile for the Streamlit app. |

---

## Known limitations / ideas for next steps

* `silhouette_score` is O(n²) memory; a chunked variant using `chunked_apply`
  would extend it to larger datasets.
* `power_iteration` finds only the dominant eigenpair; deflation or Lanczos
  would be natural follow-ups.
* An LU decomposition with partial pivoting, DBSCAN and a Gaussian-mixture EM
  would round out the "from scratch" collection.
* The notebooks still use the original function signatures; a follow-up pass
  could showcase `KMeans`, `pca`, `RunningStats` and `timeit_compare`.
