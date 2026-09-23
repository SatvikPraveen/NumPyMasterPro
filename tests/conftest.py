"""
Pytest configuration and shared fixtures for NumPyMasterPro tests
"""

import os

import numpy as np
import pytest

# Streamlit (used by tests/test_app.py) POSTs usage statistics on every script
# run; without network access that call blocks for ~20 s per run. Opt out
# before streamlit can be imported anywhere in the session.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


@pytest.fixture
def sample_1d_array():
    """Fixture providing a simple 1D array"""
    return np.array([1, 2, 3, 4, 5])


@pytest.fixture
def sample_2d_array():
    """Fixture providing a simple 2D array"""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def random_array():
    """Fixture providing a random array from a fixed-seed Generator"""
    return np.random.default_rng(42).random((10, 5))


@pytest.fixture
def array_with_nans():
    """Fixture providing an array with NaN values"""
    return np.array([1, 2, np.nan, 4, np.nan, 6])


@pytest.fixture
def array_with_infs():
    """Fixture providing an array with infinite values"""
    return np.array([1, 2, np.inf, 4, -np.inf, 6])


@pytest.fixture
def clustering_data():
    """Fixture providing three well-separated 2-D Gaussian blobs (90 x 2)"""
    rng = np.random.default_rng(42)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 0.0]])
    return np.vstack([rng.normal(loc=c, size=(30, 2)) for c in centers])


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
