.PHONY: help install install-dev test test-verbose test-coverage lint format clean docker-build docker-run notebooks

help:  ## Show this help message
	@echo "NumPyMasterPro - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies (includes testing tools)
	pip install -r requirements.txt
	pip install -r requirements_dev.txt

test:  ## Run all tests
	pytest tests/ -v

test-verbose:  ## Run tests with verbose output
	pytest tests/ -vv -s

test-coverage:  ## Run tests with coverage report
	pytest tests/ -v --cov=scripts --cov-report=term-missing --cov-report=html

test-specific:  ## Run specific test file (use TEST=test_file_name)
	pytest tests/test_$(TEST).py -v

lint:  ## Run linting checks (flake8)
	flake8 scripts/ tests/ --max-line-length=127 --extend-ignore=E203,W503

format:  ## Format code with black and isort
	black scripts/ tests/
	isort scripts/ tests/

format-check:  ## Check code formatting without changes
	black --check scripts/ tests/
	isort --check-only scripts/ tests/

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ coverage.xml .mypy_cache/

docker-build:  ## Build Docker image
	docker-compose build

docker-run:  ## Run Docker container with Jupyter
	docker-compose up

docker-down:  ## Stop Docker containers
	docker-compose down

notebooks:  ## Start Jupyter Lab locally
	jupyter lab

streamlit:  ## Run Streamlit K-Means app
	streamlit run kmeans_app.py

venv:  ## Create virtual environment
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

setup: venv install  ## Complete setup (venv + dependencies)
	@echo "Setup complete! Activate venv: source venv/bin/activate"

all: clean install-dev test lint  ## Run all checks (clean, install, test, lint)
