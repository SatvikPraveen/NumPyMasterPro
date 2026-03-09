"""
Pytest configuration and shared fixtures for NumPyMasterPro tests
"""
import pytest
import numpy as np


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
    """Fixture providing a random array with fixed seed"""
    np.random.seed(42)
    return np.random.rand(10, 5)


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
    """Fixture providing simple clustering data"""
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    cluster3 = np.random.randn(30, 2) + [5, 0]
    return np.vstack([cluster1, cluster2, cluster3])


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
