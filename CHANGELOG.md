# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-09-23

A ground-up hardening of the utility package. The notebook-facing API is
unchanged (every function the notebooks call keeps its name and positional
signature), but internals, numerics and tooling are new.

### Added
- **K-Means**: `k-means++` seeding, `n_init` restarts, `run_kmeans` returning a
  `KMeansResult` (centroids, labels, inertia, `n_iter`, `converged`,
  `inertia_history`), `pairwise_sq_distances`, `silhouette_score`,
  `elbow_point`, and a scikit-learn-style `KMeans` class with
  `fit`/`predict`/`transform`/`score`.
- **Linear algebra**: `least_squares` (lstsq / QR / Cholesky / normal),
  `ridge_regression`, `forward_substitution`, `back_substitution`,
  `cholesky_solve`, `qr_solve`, `gram_schmidt` (modified), `power_iteration`,
  `pca` (+ `PCAResult.inverse_transform`), `condition_number`, `is_symmetric`,
  `is_positive_definite`, `frobenius_norm`, `project_onto`.
- **Statistics**: `RunningStats` (Welford update, Chan merge), `bootstrap_ci`,
  `pearson_r`, `covariance_matrix`, `weighted_mean`/`weighted_var`/`weighted_std`,
  `ecdf`, `entropy`, `moving_average`; `summarize_array` reports count, skewness
  and excess kurtosis and accepts `nan_policy`.
- **Arrays**: `broadcast_result_shape`, `explain_broadcast`, `is_view_of`,
  `human_bytes`; `describe_array` reports strides and view status;
  `compare_arrays` reports `allclose` and `shares_memory`.
- **Performance** (new module `perf_utils`): `Timer`, `timeit_compare`,
  `memory_footprint`, `rolling_windows`, `strided_blocks`, `iter_chunks`,
  `chunked_apply`.
- **I/O**: compressed `.npz`, `as_dict` loading, `mmap_mode`, `filling_values`,
  explicit memmap fills; all functions accept `str` or `Path`, create parent
  directories and return the written path.
- **Streamlit app**: sidebar controls, PCA projection for >2 features,
  convergence and elbow plots, silhouette metric, labelled CSV export.
- **Tooling**: `pyproject.toml` (metadata, ruff, mypy, pytest, coverage),
  pre-commit config, `hypothesis` property-based tests, headless app tests via
  `streamlit.testing`, `Makefile` targets for every check.
- **CI**: lint (ruff + mypy) gate, Python 3.10–3.13 matrix, notebook execution
  that actually fails on error, bandit + pip-audit, Docker build with layer
  cache and in-image smoke test.
- **Docker**: multi-stage `python:3.12-slim` image using uv, unprivileged user,
  healthcheck, `app` compose profile for Streamlit.

### Changed
- All randomness uses `numpy.random.Generator`; functions take `seed=` (int,
  Generator or None) instead of mutating global `np.random` state.
- `kmeans()` default initialisation is now `k-means++`; labels are recomputed
  against the final centroids so results are self-consistent.
- `closed_form_linear_regression` defaults to SVD-based `lstsq`; the historical
  `inv(XᵀX)Xᵀy` path is available as `method="normal"`.
- Normalisers (`minmax_normalize`, `zscore_normalize`, `robust_scale`) gained
  an `axis` argument and return zeros for constant slices instead of NaN/inf.
- `array_flags` reads `ndarray.flags` attributes instead of parsing its repr.
- `mask_by_value` dispatches through the `operator` module instead of eagerly
  evaluating all comparisons.
- `requirements_dev.txt` is a curated list rather than a `pip freeze` dump;
  `pytest.ini` was folded into `pyproject.toml`.
- Docker base image moved off the deprecated `jupyter/minimal-notebook`.

### Fixed
- `update_centroids` no longer produces NaN centroids for empty clusters.
- `entropy` could return NaN when a tiny count underflowed to zero after
  normalisation.
- `zscore_normalize` returned z-scores of 1 for identical tiny values whose
  mean rounds by one ulp.
- `mask_by_value` evaluated every comparison even when only one was requested.
- The clustering test fixture had been mangled into a ragged list by an
  earlier automated refactor.

### Removed
- flake8 / black / isort configuration (replaced by ruff).
- The retired `safety check` CI step (replaced by pip-audit).

## [1.0.0] - 2026-03-09

Initial release: ten notebooks, utility scripts, Streamlit demo, pytest suite,
GitHub Actions pipeline, Docker setup.
