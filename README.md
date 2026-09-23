# 🧠 NumPyMasterPro

[![CI](https://github.com/SatvikPraveen/NumPyMasterPro/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/NumPyMasterPro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-darkgreen.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A5%201.26-013243.svg?logo=numpy)](https://numpy.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20hypothesis-blue.svg)](https://docs.pytest.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blueviolet.svg)](https://www.docker.com/)

**NumPyMasterPro** is a hands-on, from-first-principles NumPy project: ten themed
notebooks, a typed and tested utility package, and algorithms implemented with
NumPy only (k-means++, PCA, ridge regression, Welford statistics, stride tricks),
wrapped in modern Python tooling.

It is meant to be three things at once: a **learning path**, a **reference
toolkit** you can drop into other projects, and a **portfolio-quality codebase**
with CI, type checking, property-based tests and a Docker image.

---

## ✨ What's inside

| Area | Highlights |
| --- | --- |
| **K-Means from scratch** | k-means++ seeding, restarts (`n_init`), empty-cluster repair, O(n·k) distance computation via ‖x‖²−2x·c+‖c‖², silhouette score, elbow detection, a scikit-learn-style `KMeans` class |
| **Linear algebra** | lstsq / QR / Cholesky least squares, ridge regression, forward & back substitution, modified Gram-Schmidt, power iteration, PCA via SVD, condition numbers |
| **Statistics** | Axis-aware, division-safe normalisers, skew/kurtosis, `RunningStats` (Welford + Chan merge), bootstrap CIs, weighted stats, moving averages, ECDF, entropy |
| **Arrays & memory** | View vs. copy detection, memory flags, broadcasting explained axis by axis, human-readable sizes |
| **Performance** | `timeit_compare` (checks outputs agree), zero-copy `rolling_windows` / `strided_blocks`, chunked processing |
| **I/O** | `.npy`/`.npz` (optionally compressed), delimited text with missing values, memory maps; all path-like aware |
| **App** | Streamlit K-Means explorer with PCA projection, elbow curve, silhouette and CSV export |

Every public function is type-annotated, documented, and covered by unit tests
plus [hypothesis](https://hypothesis.readthedocs.io/) property-based tests
(coverage ≈ 96%). Randomness flows through `numpy.random.Generator` everywhere,
so results are reproducible with a `seed=` argument and no global state.

---

## 🚀 Quick start

```bash
git clone https://github.com/SatvikPraveen/NumPyMasterPro.git
cd NumPyMasterPro

# Option A: uv (fast)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,app,notebooks]"

# Option B: pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app,notebooks]"

make check          # ruff + mypy + full test suite
make notebooks      # Jupyter Lab
make streamlit      # K-Means explorer at http://localhost:8501
```

Only `numpy` and `pandas` are required at runtime; `app`, `notebooks` and `dev`
are optional extras (see `pyproject.toml`).

### 30-second tour

```python
import numpy as np
from scripts import KMeans, generate_data, silhouette_score, elbow_point, compute_inertia
from scripts.linear_algebra_utils import pca, ridge_regression, closed_form_linear_regression
from scripts.stats_utils import RunningStats, bootstrap_ci, zscore_normalize
from scripts.perf_utils import timeit_compare, rolling_windows

X = generate_data(n_per_cluster=100, seed=0)          # 3 Gaussian blobs, (300, 2)

km = KMeans(n_clusters=3, seed=0).fit(X)               # k-means++, 10 restarts
km.inertia_, km.n_iter_, silhouette_score(X, km.labels_)

ks = range(1, 9)
elbow_point(list(ks), compute_inertia(X, ks, seed=0))  # -> 3

w = closed_form_linear_regression(X[:, :1], X[:, 1])   # SVD-based lstsq; w[0] is the intercept
w_ridge = ridge_regression(X[:, :1], X[:, 1], alpha=1.0)

res = pca(X, n_components=1)                           # components, explained_variance_ratio, transformed

rs = RunningStats()
for chunk in np.array_split(X[:, 0], 10):              # stream data without a second pass
    rs.update(chunk)
rs.mean, rs.std()

bootstrap_ci(X[:, 0], np.median, seed=0)               # percentile bootstrap, fully vectorised

timeit_compare({"vectorised": lambda a: a * 2,
                "loop": lambda a: np.array([v * 2 for v in a])}, X[:, 0])
```

---

## 🧱 Project layout

```
NumPyMasterPro/
├── notebooks/                # 📓 10 themed notebooks (basics → K-Means from scratch)
├── scripts/                  # 🛠️ Installable package `numpymasterpro` (import as `scripts`)
│   ├── array_utils.py        #    metadata, views vs copies, broadcasting diagnostics
│   ├── linear_algebra_utils.py  # solvers, factorisations, regression, PCA
│   ├── stats_utils.py        #    normalisers, online stats, bootstrap, moving averages
│   ├── kmeans_utils.py       #    k-means++ / Lloyd / silhouette / elbow / KMeans class
│   ├── perf_utils.py         #    timing, stride tricks, chunked processing
│   ├── io_utils.py           #    npy / npz / txt / memmap
│   ├── math_utils.py, aggregation_utils.py, logical_utils.py
├── tests/                    # 🧪 unit tests, hypothesis property tests, headless app tests
├── datasets/                 # 📁 sample data used by the notebooks
├── docs/                     # 📜 cheat sheet, testing guide, implementation notes
├── kmeans_app.py             # 🎛️ Streamlit explorer
├── pyproject.toml            # 📦 metadata + ruff / mypy / pytest / coverage config
├── Dockerfile, docker-compose.yml, Makefile, .pre-commit-config.yaml
└── .github/workflows/ci.yml  # 🤖 lint → tests (4 Pythons × 3 OSes) → notebooks → security → docker
```

---

## 🧮 Notebooks

| Notebook | Description |
| --- | --- |
| `01_array_basics.ipynb` | Array creation, dtypes, shapes, memory attributes |
| `02_indexing_slicing.ipynb` | Indexing, slicing, masking, `.take()`, `.put()` |
| `03_array_manipulation.ipynb` | Reshaping, stacking, splitting, tiling, padding |
| `04_math_operations.ipynb` | Element-wise ops, aggregation, rounding, broadcasting |
| `05_linear_algebra.ipynb` | Products, inverse, norms, eig/SVD, solving systems |
| `06_statistics_probability.ipynb` | Descriptive stats, histograms, correlation, sampling |
| `07_masking_conditions.ipynb` | `where`, `select`, logical ops, `nonzero`, `isfinite` |
| `08_file_io_memory.ipynb` | `save`/`load`, memmap, vectorize, views vs. copies |
| `09_real_world_cases.ipynb` | Regression, image ops, time-series scaling, simulation |
| `10_kmeans_from_scratch.ipynb` | 🎯 K-Means + Elbow Method using NumPy only |

Notebooks import the utilities directly (`from kmeans_utils import kmeans`), and
CI executes every notebook top-to-bottom on each push.

---

## 🧪 Quality gates

```bash
make lint           # ruff check
make format         # ruff --fix + ruff format
make typecheck      # mypy (strict-ish, no implicit Optional)
make test           # pytest with branch coverage
make test-props     # hypothesis property tests only
make notebooks-exec # execute all notebooks headlessly
make precommit      # install pre-commit hooks
```

The CI pipeline (`.github/workflows/ci.yml`) runs lint → tests on Python
3.10–3.13 across Ubuntu/macOS/Windows → notebook execution → bandit + pip-audit
→ Docker build & smoke test. See [docs/TESTING.md](docs/TESTING.md) for the
testing guide.

---

## 🐳 Docker

```bash
docker compose up --build            # Jupyter Lab  → http://localhost:8889
docker compose --profile app up app  # Streamlit    → http://localhost:8501
docker compose --profile app down
```

The image is a multi-stage `python:3.12-slim` build with dependencies resolved by
uv, runs as an unprivileged user, and reads `JUPYTER_TOKEN` / `JUPYTER_PASSWORD`
from the environment (empty by default for a login-free local lab). Copy
`.env.example` to `.env` to override ports and credentials.

---

## 🎛️ Streamlit K-Means explorer

```bash
streamlit run kmeans_app.py
```

Pick demo blobs or upload a CSV, choose features (more than two are projected
with the project's own PCA), tune `k`, initialisation, restarts and seed, then
read off inertia, iterations, silhouette, the convergence curve and an elbow
plot with a suggested `k`. Export the labelled rows as CSV.

---

## 🤝 Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Run `make check` before opening a PR.

## 📄 License

GNU General Public License v3.0 — see [LICENSE](./LICENSE).

## 🌟 Showcase & Star

If this project helped you master NumPy, feel free to ⭐ it and share it with others!
