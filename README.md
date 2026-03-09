# 🧠 NumPyMasterPro

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-darkgreen.svg)](https://www.python.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen.svg)](https://github.com/SatvikPraveen/NumPyMasterPro/actions)
[![Tests](https://img.shields.io/badge/Tests-Pytest-blue.svg)](https://docs.pytest.org/)
[![Issues](https://img.shields.io/github/issues/SatvikPraveen/NumPyMasterPro?color=yellowgreen)](https://github.com/SatvikPraveen/NumPyMasterPro/issues)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blueviolet.svg)](https://www.docker.com/)
[![NumPy Focused](https://img.shields.io/badge/NumPy-100%25-brightgreen.svg)](https://numpy.org/)
[![Real-World Use Cases](https://img.shields.io/badge/Use%20Cases-Included-ff69b4.svg)](#)
[![K-Means Project](https://img.shields.io/badge/Project-K--Means%20From%20Scratch-9cf.svg)](#)

**NumPyMasterPro** is a comprehensive, modular, and hands-on project designed to help you **master NumPy from first principles to real-world applications**.

This project isn't just a learning exercise — it's a **complete reference toolkit**, **interview-ready resource**, and a **portfolio-quality project** that showcases your fluency with one of Python’s most essential libraries for scientific computing and data analysis.

---

## 🚀 Why This Project Matters

> Most learners stop at tutorials. This repository takes you further — by combining theory, implementation, real-world use cases, and production practices in one place.

✅ Covers **100% of NumPy's essential concepts**  
✅ Demonstrates **clean project structure and modular code reuse**  
✅ Includes **interview-ready topics** like broadcasting, vectorization, and matrix algebra  
✅ Provides **Jupyter notebooks + Python utility scripts + cheat sheet**  
✅ Ends with a **K-Means algorithm from scratch with Elbow Method** — great for resumes

---

## 📌 Project Objectives

- 🔍 **Master Core NumPy Syntax** through progressively organized notebooks
- 🔄 **Understand Memory Efficiency**: broadcasting, vectorization, views vs. copies
- ⚙️ **Practice Clean Coding** using reusable utility scripts in `/scripts`
- 🧠 **Explore Real-World Scenarios**: regression, simulations, image ops, clustering
- 📂 **Build a Reference Toolkit** for revision, projects, and technical interviews

---

## 🧱 Folder Structure

```bash
NumPyMasterPro/
├── notebooks/                 # 📓 Themed Jupyter Notebooks (core + advanced topics)
├── scripts/                   # 🛠️ Modular, reusable Python utilities
├── datasets/                  # 📁 Data files used in notebooks
├── docs/                      # 📜 Cheat sheets and markdown-based quick notes
├── requirements.txt           # 📦 Minimal dependencies to run the project
├── requirements_dev.txt       # 📦 Full dev environment
├── .env.example               # 🛡️ Sample env file for Docker-based config (login-free setup)
├── docker-compose.yml         # 🐳 Multi-container orchestration for Jupyter Lab
├── Dockerfile                 # 🐳 Docker image setup using Jupyter minimal notebook base
├── .gitignore                 # ❌ Files to exclude from version control
└── README.md                  # 📘 This file!
```

---

## 🧮 Topics Covered

| Notebook                          | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `01_array_basics.ipynb`           | Array creation, types, shapes, memory attributes             |
| `02_indexing_slicing.ipynb`       | Indexing, slicing, masking, `.take()`, `.put()`              |
| `03_array_manipulation.ipynb`     | Reshaping, stacking, splitting, tiling, padding              |
| `04_math_operations.ipynb`        | Element-wise ops, aggregation, rounding, broadcasting        |
| `05_linear_algebra.ipynb`         | Dot product, inverse, norms, eig/SVD, solving systems        |
| `06_statistics_probability.ipynb` | Descriptive stats, histograms, correlations, sampling        |
| `07_masking_conditions.ipynb`     | `where`, `select`, logical ops, `nonzero`, `isfinite`, etc.  |
| `08_file_io_memory.ipynb`         | `save`, `load`, `memmap`, vectorize, views vs. copies        |
| `09_real_world_cases.ipynb`       | Regression, image ops, time-series scaling, simulations      |
| `10_kmeans_from_scratch.ipynb`    | 🎯 BONUS: K-Means Clustering + Elbow Method using NumPy only |

---

## 🧰 Utility Scripts

| File                      | Purpose                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `array_utils.py`          | Inspect shapes, types, identities, and metadata                |
| `linear_algebra_utils.py` | Matrix algebra: dot, inverse, SVD, eigenvalues                 |
| `math_utils.py`           | Element-wise math: power, root, trig, rounding, logs, exponent |
| `aggregation_utils.py`    | Sum, mean, std, var, min, max — global & axis-wise             |
| `stats_utils.py`          | Z-score, normalization, correlation, histogram bins            |
| `logical_utils.py`        | Boolean logic, masking, conditionals (`any`, `all`, `where`)   |
| `kmeans_utils.py`         | K-Means from scratch, inertia calculation, and centroid init   |

Example usage:

```python
# Direct module import
from scripts.kmeans_utils import kmeans, compute_inertia

# Or use convenient re-exports from __init__.py
from scripts import kmeans, describe_array, minmax_normalize
```

---

## 🎛️ Streamlit Frontend (Interactive Demo)

You can try the K-Means algorithm with different datasets or number of clusters using:

```bash
streamlit run kmeans_app.py
```

This allows you to upload `.csv` files, set cluster count, and visualize results in real time.
Great for experiments, education, and showcasing clustering interactively.

---

## 🐳 Docker-Based Setup (Optional)

Prefer running in a **containerized Jupyter Lab** environment?

```bash
docker compose up --build
```

Then open the browser at:
👉 [http://localhost:8889](http://localhost:8889)

> You can also stop the container with:

```bash
docker compose down --volumes --remove-orphans
```

---

## 🔐 Authentication & Security

This project is configured for **login-free use** of Jupyter Lab — no password or token required.

- ✅ `.env.example` is included with recommended settings.
- 🚫 `.env` is deliberately **excluded** from the repo (add your own if needed).
- 🛡️ You may modify the `docker-compose.yml` to add a token or hashed password later.

---

## 🧠 Recommended Use

- ✍️ Study each notebook sequentially and refer back as needed
- 🧪 Use `/scripts/` functions in other projects or interview tasks
- 🧵 Treat `docs/numpy_cheatsheet.md` as your quick review guide
- 🧠 Use `10_kmeans_from_scratch.ipynb` in your resume to show NumPy fluency
- 💡 Add your own notebooks (e.g., PCA from scratch, numerical integration, etc.)

---

## 🔧 Getting Started (Without Docker)

1. **Clone the repo**

   ```bash
   git clone https://github.com/SatvikPraveen/NumPyMasterPro.git
   cd NumPyMasterPro
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the Jupyter Lab interface**

   ```bash
   jupyter lab
   ```

---

## 🧪 Testing

**NumPyMasterPro** includes a comprehensive test suite with **80+ unit tests** covering all utility modules.

### Quick Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=term-missing
```

### Using Makefile Commands

```bash
make test              # Run all tests
make test-coverage     # Generate coverage report
make lint              # Check code quality
make format            # Auto-format code
make all               # Run complete checks
```

### Test Coverage

- ✅ Array utilities (describe, compare, flags)
- ✅ Logical operations (any, all, where, masking)
- ✅ K-Means algorithm (clustering, inertia)
- ✅ Math operations (arithmetic, trig, rounding)
- ✅ Linear algebra (matrices, eigenvalues, SVD)
- ✅ Statistics (normalization, correlation)

📖 **Detailed testing guide:** [TESTING.md](./TESTING.md)

### CI/CD Pipeline

Automated testing runs on:
- 🔄 Every push to `main`/`develop`
- 🔄 All pull requests  
- ✅ Multi-OS (Ubuntu, macOS, Windows)
- ✅ Python 3.10, 3.11, 3.12
- ✅ Code linting & formatting checks
- ✅ Notebook validation
- ✅ Docker build verification

---

## 📄 License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0). See the [LICENSE](./LICENSE) file for more details.

---

## 🌟 Showcase & Star

If this project helped you master NumPy, feel free to ⭐ it and share it with others!
