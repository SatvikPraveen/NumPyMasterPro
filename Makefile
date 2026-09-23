.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test test-fast test-coverage test-props lint format typecheck check \
        precommit clean docker-build docker-run docker-app docker-down notebooks streamlit notebooks-exec all

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

help:  ## Show this help message
	@echo "NumPyMasterPro - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
venv:  ## Create a virtual environment in .venv (uses uv when available)
	@if command -v uv >/dev/null 2>&1; then uv venv $(VENV); else $(PYTHON) -m venv $(VENV); fi
	@echo "Activate with: source $(BIN)/activate"

install:  ## Install runtime dependencies (editable package)
	$(BIN)/pip install -e .

install-dev:  ## Install package with dev, app and notebook extras
	@if command -v uv >/dev/null 2>&1; then uv pip install --python $(BIN)/python -e ".[dev,app,notebooks]"; \
	 else $(BIN)/pip install -e ".[dev,app,notebooks]"; fi

setup: venv install-dev  ## Complete setup (venv + all extras)

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test:  ## Run the full test suite with coverage
	$(BIN)/pytest

test-fast:  ## Run tests in parallel without coverage
	$(BIN)/pytest -n auto --no-cov -q

test-coverage:  ## Run tests and write an HTML coverage report to htmlcov/
	$(BIN)/pytest --cov-report=html --cov-report=term-missing
	@echo "open htmlcov/index.html"

test-props:  ## Run only the hypothesis property-based tests
	$(BIN)/pytest tests/test_properties.py --no-cov -q

lint:  ## Lint with ruff
	$(BIN)/ruff check scripts tests kmeans_app.py

format:  ## Auto-fix lint issues and format with ruff
	$(BIN)/ruff check scripts tests kmeans_app.py --fix
	$(BIN)/ruff format scripts tests kmeans_app.py

format-check:  ## Verify formatting without changing files
	$(BIN)/ruff format --check scripts tests kmeans_app.py

typecheck:  ## Static type-check scripts/ with mypy
	$(BIN)/mypy

check: lint format-check typecheck test  ## Everything CI runs, locally

precommit:  ## Install git pre-commit hooks
	$(BIN)/pre-commit install

# ---------------------------------------------------------------------------
# Notebooks & app
# ---------------------------------------------------------------------------
notebooks:  ## Start Jupyter Lab locally
	$(BIN)/jupyter lab

notebooks-exec:  ## Execute every notebook headlessly (what CI does)
	@set -e; for nb in notebooks/*.ipynb; do \
	  echo "→ $$nb"; \
	  $(BIN)/jupyter nbconvert --to notebook --execute "$$nb" --ExecutePreprocessor.timeout=300 --output "/tmp/$$(basename $$nb)"; \
	done

streamlit:  ## Run the Streamlit K-Means explorer
	$(BIN)/streamlit run kmeans_app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker compose build

docker-run:  ## Run Jupyter Lab in Docker (http://localhost:8889)
	docker compose up

docker-app:  ## Run the Streamlit app in Docker (http://localhost:8501)
	docker compose --profile app up app

docker-down:  ## Stop and remove containers
	docker compose --profile app down --remove-orphans

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Remove caches, coverage and build artefacts
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" -o -name ".hypothesis" -o -name "*.egg-info" \) -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".coverage" \) -delete
	rm -rf htmlcov/ coverage.xml build/ dist/

all: clean install-dev check  ## Clean, install, and run all checks
