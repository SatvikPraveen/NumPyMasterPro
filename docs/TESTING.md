# 🧪 Testing Guide — NumPyMasterPro

This document describes the test suite, the quality gates that run locally and
in CI, and how to add tests for new code.

---

## 📋 Overview

The suite has three layers:

| Layer | Files | What it checks |
| --- | --- | --- |
| **Unit tests** | `tests/test_*_utils.py` | Every public function on hand-picked inputs, edge cases and error paths, cross-checked against `numpy.linalg` / closed-form results where possible |
| **Property-based tests** | `tests/test_properties.py` | Invariants on generated inputs via [hypothesis](https://hypothesis.readthedocs.io/): e.g. inertia never increases across Lloyd iterations, `RunningStats` equals a single NumPy pass for any chunking, PCA round-trips |
| **Headless app tests** | `tests/test_app.py` | The Streamlit explorer via `streamlit.testing.v1.AppTest`; skipped when the optional `app` extra is not installed |

Coverage is measured with branch coverage and sits around **96%** of `scripts/`.

---

## 🚀 Quick start

```bash
uv pip install -e ".[dev,app]"     # or: pip install -e ".[dev,app]"

pytest                             # full suite with coverage (configured in pyproject.toml)
pytest -n auto --no-cov -q         # fast parallel run
pytest tests/test_properties.py    # only the property-based tests
pytest -k "silhouette" -v          # tests matching a keyword
pytest -m "not integration"        # skip the app tests
```

Or via the Makefile:

```bash
make test            # pytest with coverage
make test-fast       # parallel, no coverage
make test-coverage   # HTML report in htmlcov/
make test-props      # hypothesis tests only
make check           # ruff + format check + mypy + tests (what CI runs)
```

---

## ⚙️ Configuration

All test configuration lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --tb=short --cov=scripts --cov-report=term-missing"
filterwarnings = ["error::DeprecationWarning:scripts"]   # our own deprecations are errors
markers = ["slow", "integration", "unit"]

[tool.coverage.run]
source = ["scripts"]
branch = true
```

`--strict-markers` means an unknown `@pytest.mark.<name>` is an error; register
new markers in the `markers` list.

---

## 🧩 Shared fixtures (`tests/conftest.py`)

- `sample_1d_array`, `sample_2d_array` — small deterministic arrays
- `random_array` — `(10, 5)` uniform values from `default_rng(42)`
- `array_with_nans`, `array_with_infs` — for NaN/inf handling paths
- `clustering_data` — three well-separated Gaussian blobs, `(90, 2)`

Fixtures use `numpy.random.Generator`; never call `np.random.seed` in tests
(ruff rule `NPY002` will flag it).

---

## 🔬 Property-based tests

`tests/test_properties.py` uses `hypothesis.extra.numpy` strategies to generate
arrays and shapes. Each test is decorated with a shared `settings` object
(`max_examples=60`, no deadline). When hypothesis finds a failing input it
prints a minimal *falsifying example*; two real bugs (an `entropy` underflow and
a z-score edge case) were found this way and are pinned in the changelog.

Guidelines:

- Bound floats (`min_value`/`max_value`, no NaN/inf) unless NaN handling is the
  point of the test.
- Compare with tolerances scaled to the data magnitude.
- Keep example counts modest; property tests run on every CI matrix leg.

---

## 🎛️ App tests

`tests/test_app.py` runs `kmeans_app.py` headlessly:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("kmeans_app.py").run()
assert not at.exception
```

It checks that the page renders without exceptions, that the metrics and
download button exist, and that changing `k` re-runs cleanly. The module calls
`pytest.importorskip("streamlit")`, so the tests vanish rather than fail when
Streamlit is absent.

---

## 🏷️ Markers

```python
@pytest.mark.slow          # long-running; deselect with -m "not slow"
@pytest.mark.integration   # crosses module boundaries or drives the app
@pytest.mark.unit          # isolated function behaviour
```

---

## 🤖 Continuous integration

`.github/workflows/ci.yml` runs on pushes and PRs to `main`/`develop`:

1. **lint** — `ruff check`, `ruff format --check`, `mypy` (gates everything else)
2. **test** — Python 3.10/3.11/3.12/3.13 on Ubuntu; 3.12/3.13 on macOS and Windows; coverage uploaded from one leg
3. **notebooks** — every notebook executed with `nbconvert`; a failing cell fails the job
4. **security** — `bandit -ll` (blocking) and `pip-audit` (advisory)
5. **docker** — image build with layer cache, then an in-container smoke test of the package
6. **build-status** — fails unless all of the above succeeded

Nothing is masked with `|| echo`; a red job means something is actually broken.

---

## ✍️ Writing new tests

1. Put unit tests in `tests/test_<module>.py`, grouped in `Test<Feature>` classes.
2. Test the happy path, at least one edge case (empty, constant, NaN), and the
   error path with `pytest.raises(ValueError, match=...)`.
3. Where a NumPy reference exists (`np.linalg.solve`, `np.cov`, `np.convolve`),
   assert against it with `np.testing.assert_allclose`.
4. If the function has an invariant that holds for *all* inputs, add a
   hypothesis test to `tests/test_properties.py`.
5. Run `make check` before pushing.

---

## 🐛 Debugging

```bash
pytest -x --lf              # stop at first failure, rerun last failures
pytest -vv -s               # verbose, show prints
pytest --pdb                # drop into the debugger on failure
pytest -o faulthandler_timeout=30   # dump stacks if a test hangs
```

---

## 📚 References

- [pytest](https://docs.pytest.org/) · [hypothesis](https://hypothesis.readthedocs.io/) · [coverage.py](https://coverage.readthedocs.io/)
- [NumPy testing guidelines](https://numpy.org/doc/stable/reference/testing.html)
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
