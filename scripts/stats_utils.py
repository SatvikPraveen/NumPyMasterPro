"""
stats_utils.py

A robust collection of statistical and random sampling utilities using NumPy.
Designed for both exploratory analysis and educational pipelines.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

import numpy as np
from typing import Dict, Tuple


# ✅ 1. Descriptive Statistics
def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    """Returns core descriptive statistics of a 1D array."""
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "var": np.var(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "range": np.ptp(arr),
        "25th_percentile": np.percentile(arr, 25),
        "75th_percentile": np.percentile(arr, 75),
        "IQR": np.percentile(arr, 75) - np.percentile(arr, 25),
    }


def quantiles(arr: np.ndarray, q: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> np.ndarray:
    """Returns quantiles (default: Q1, median, Q3)."""
    return np.quantile(arr, q)


# ✅ 2. Correlation & Covariance
def compute_covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the 2x2 covariance matrix of two arrays."""
    return np.cov(x, y)

def compute_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the 2x2 correlation coefficient matrix."""
    return np.corrcoef(x, y)


# ✅ 3. Normalization Utilities
def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Performs min-max normalization to scale data between 0 and 1."""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Applies z-score standardization (mean=0, std=1)."""
    return (arr - np.mean(arr)) / np.std(arr)

def robust_scale(arr: np.ndarray) -> np.ndarray:
    """Scales using the interquartile range (IQR) — robust to outliers."""
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return (arr - q1) / iqr if iqr != 0 else arr - q1


# ✅ 4. Random Sampling Utilities
def generate_random_integers(low: int, high: int, size: int, seed: int = None) -> np.ndarray:
    """Generates random integers in the range [low, high)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(low, high, size)

def generate_random_floats(size: int, seed: int = None) -> np.ndarray:
    """Generates uniform random floats in [0,1)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(size)

def generate_normal_distribution(size: int, seed: int = None) -> np.ndarray:
    """Generates standard normal random values."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(size)

def random_sample(arr: np.ndarray, sample_size: int, replace: bool = True) -> np.ndarray:
    """Samples elements from a 1D array."""
    return np.random.choice(arr, size=sample_size, replace=replace)


# ✅ 5. Histogram & Bincount
def histogram_binning(arr: np.ndarray, bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Returns histogram counts and bin edges."""
    return np.histogram(arr, bins=bins)

def compute_bincount(arr: np.ndarray) -> np.ndarray:
    """Returns the count of each non-negative integer value."""
    return np.bincount(arr)

def standardize_rows(arr):
    """
    Standardizes each row of a 2D NumPy array to mean=0, std=1.
    """
    means = arr.mean(axis=1, keepdims=True)
    stds = arr.std(axis=1, keepdims=True)
    return (arr - means) / stds