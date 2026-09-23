# 🤝 Contributing to NumPyMasterPro

Thank you for your interest in contributing to **NumPyMasterPro**! 🚀  
Whether you're fixing a bug, adding a notebook, improving documentation, or sharing a use case — your contribution is highly appreciated. 💙

---

## 🧭 Project Structure (Quick Overview)

```

NumPyMasterPro/
├── notebooks/                 # Jupyter Notebooks for learning NumPy
├── scripts/                   # Installable utility package (typed, tested)
├── tests/                     # pytest unit tests, hypothesis property tests, app tests
├── datasets/                  # Sample data files
├── docs/                      # Cheat sheet, testing guide, implementation notes
├── pyproject.toml             # Metadata + ruff / mypy / pytest configuration
├── kmeans_app.py              # Streamlit K-Means explorer
└── README.md

````

---

## ✅ How to Contribute

### 1. Fork the Repo

Click the **Fork** button at the top right of this page, then clone your fork:

```bash
git clone https://github.com/SatvikPraveen/NumPyMasterPro.git
cd NumPyMasterPro
```

### 2. Create a Branch

Create a new branch for your contribution:

```bash
git checkout -b feature/my-awesome-idea
```

### 3. Set Up a Development Environment

```bash
uv venv .venv && source .venv/bin/activate      # or: python -m venv .venv
uv pip install -e ".[dev,app,notebooks]"        # or: pip install -e ".[dev,app,notebooks]"
make precommit                                  # installs ruff / mypy git hooks
```

### 4. Make Your Changes

* Every public function in `scripts/` needs a docstring, type hints and tests.
* Use `numpy.random.Generator` with a `seed=` argument; never `np.random.seed`.
* Validate inputs and raise `ValueError` with a clear message instead of
  letting NumPy emit NaN or a cryptic error.
* Keep the notebook-facing function names and positional signatures stable.
* Add a line to `CHANGELOG.md` under *Unreleased*.

### 5. Run the Quality Gates

```bash
make check          # ruff check + ruff format --check + mypy + pytest
make test-props     # hypothesis property tests only
make notebooks-exec # execute notebooks headlessly (what CI does)
```

CI runs the same checks on Python 3.10–3.13 across three operating systems,
executes every notebook, scans with bandit, and builds the Docker image.
A pull request must be green before review.

### 6. Commit & Push

```bash
git add .
git commit -m "feat(stats): add trimmed_mean with tests"   # conventional-commit style
git push origin feature/my-awesome-idea
```

### 7. Submit a Pull Request

* Go to your fork on GitHub
* Click **"Compare & pull request"**
* Add a clear description of your change

---

## 💡 Contribution Ideas

* Add new notebooks (e.g., numerical integration, NumPy random tips, simulation)
* Add explanations to existing notebooks
* Extend `scripts/` (ideas: LU decomposition with pivoting, DBSCAN, Gaussian
  mixture EM, FFT-based convolution, `einsum` walkthroughs)
* Add hypothesis property tests for existing invariants
* Improve `docs/` cheat sheets
* Add datasets for experimentation
* Report or fix bugs

---

## 📜 Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## 🙌 Thank You!

Your contributions help grow this project and the community around it.
Let’s build NumPy fluency — together!

— *The NumPyMasterPro Maintainers*

```
