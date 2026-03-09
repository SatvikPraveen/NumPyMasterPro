# 🧪 Testing Guide - NumPyMasterPro

This document describes the testing infrastructure and best practices for NumPyMasterPro.

---

## 📋 Overview

NumPyMasterPro includes a comprehensive test suite covering all utility modules with **unit tests**, **integration tests**, and **CI/CD automation**.

### Test Coverage

- ✅ **Array Utilities** - `test_array_utils.py`
- ✅ **Logical Utilities** - `test_logical_utils.py`
- ✅ **K-Means Utilities** - `test_kmeans_utils.py`
- ✅ **Math Utilities** - `test_math_utils.py`
- 🔄 Additional modules can be tested similarly

---

## 🚀 Quick Start

### Prerequisites

Install development dependencies:

```bash
pip install -r requirements_dev.txt
```

Or use the Makefile:

```bash
make install-dev
```

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run tests with coverage:**
```bash
pytest --cov=scripts --cov-report=term-missing
```

**Run specific test file:**
```bash
pytest tests/test_logical_utils.py -v
```

**Run tests matching a pattern:**
```bash
pytest -k "test_any_condition" -v
```

---

## 📊 Using Makefile Commands

We provide convenient Makefile commands for common tasks:

```bash
make test              # Run all tests
make test-coverage     # Run tests with HTML coverage report
make test-verbose      # Run tests with detailed output
make lint              # Check code quality
make format            # Auto-format code with black & isort
make clean             # Remove cache and build artifacts
make all               # Run complete check (clean, install, test, lint)
```

---

## 🎯 Test Structure

### Directory Layout

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_array_utils.py      # Tests for array utilities
├── test_logical_utils.py    # Tests for logical operations
├── test_kmeans_utils.py     # Tests for K-Means algorithm
└── test_math_utils.py       # Tests for math operations
```

### Test Organization

Each test file follows this structure:

```python
class TestFeatureName:
    """Group related tests together"""
    
    def test_basic_case(self):
        """Test description"""
        # Arrange
        arr = np.array([1, 2, 3])
        
        # Act
        result = function_under_test(arr)
        
        # Assert
        assert result == expected_value
```

---

## 🧩 Shared Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_1d_array` - Simple 1D array
- `sample_2d_array` - Simple 2D array  
- `random_array` - Random array with fixed seed
- `array_with_nans` - Array containing NaN values
- `array_with_infs` - Array with infinite values
- `clustering_data` - Synthetic clustering dataset

**Usage:**
```python
def test_with_fixture(sample_1d_array):
    result = some_function(sample_1d_array)
    assert result.shape == (5,)
```

---

## 📈 Coverage Reports

### Terminal Coverage

```bash
pytest --cov=scripts --cov-report=term-missing
```

### HTML Coverage Report

```bash
pytest --cov=scripts --cov-report=html
open htmlcov/index.html  # View in browser
```

### XML Coverage (for CI/CD)

```bash
pytest --cov=scripts --cov-report=xml
```

---

## 🏷️ Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (takes time)"""
    pass

@pytest.mark.unit
def test_single_function():
    """Unit test for isolated function"""
    pass

@pytest.mark.integration
def test_workflow():
    """Integration test across modules"""
    pass
```

**Run tests by marker:**
```bash
pytest -m "not slow"      # Skip slow tests
pytest -m "unit"          # Run only unit tests
pytest -m "integration"   # Run only integration tests
```

---

## 🔧 Configuration

Test configuration is defined in `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -v --strict-markers --cov=scripts
```

---

## 🤖 Continuous Integration

Tests run automatically via GitHub Actions on:
- ✅ Push to `main` or `develop` branches
- ✅ Pull requests
- ✅ Manual workflow dispatch

### CI Workflow Includes:

1. **Multi-platform testing** (Ubuntu, macOS, Windows)
2. **Python version matrix** (3.10, 3.11, 3.12)
3. **Code linting** (flake8, black, isort)
4. **Notebook validation**
5. **Docker build verification**
6. **Security scanning** (safety, bandit)
7. **Coverage reporting** (Codecov)

---

## ✍️ Writing New Tests

### Step 1: Create Test File

```bash
touch tests/test_new_module.py
```

### Step 2: Import Module and Fixtures

```python
import pytest
import numpy as np
from scripts.new_module import function_to_test
```

### Step 3: Write Test Classes

```python
class TestNewFunction:
    def test_basic_behavior(self):
        result = function_to_test([1, 2, 3])
        assert result == expected
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            function_to_test([])
```

### Step 4: Run and Verify

```bash
pytest tests/test_new_module.py -v
```

---

## 📝 Best Practices

✅ **Test one thing per test** - Keep tests focused  
✅ **Use descriptive names** - `test_returns_empty_array_for_missing_values`  
✅ **Test edge cases** - Empty arrays, NaN, inf, negative values  
✅ **Use fixtures** - Reuse common test data  
✅ **Check both success and failure** - Use `pytest.raises()` for exceptions  
✅ **Aim for high coverage** - Target 80%+ code coverage  
✅ **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`

---

## 🐛 Debugging Tests

### Run with verbose output:
```bash
pytest -vv -s
```

### Run with pdb debugger:
```bash
pytest --pdb
```

### Show local variables on failure:
```bash
pytest -l
```

### Run last failed tests only:
```bash
pytest --lf
```

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [NumPy Testing Guidelines](https://numpy.org/doc/stable/reference/testing.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

© 2025 Satvik Praveen – _NumPyMasterPro Testing Guide_
