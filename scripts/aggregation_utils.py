"""
aggregation_utils.py

Utility functions for aggregation and reduction operations using NumPy.
Handles global and axis-specific operations like sum, mean, min, max,
standard deviation, and variance.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

import numpy as np

# ✅ Global Aggregations
def array_sum(arr: np.ndarray) -> float:
    return np.sum(arr)

def array_mean(arr: np.ndarray) -> float:
    return np.mean(arr)

def array_min(arr: np.ndarray) -> float:
    return np.min(arr)

def array_max(arr: np.ndarray) -> float:
    return np.max(arr)

def array_std(arr: np.ndarray) -> float:
    return np.std(arr)

def array_var(arr: np.ndarray) -> float:
    return np.var(arr)


# ✅ Axis-wise Aggregations
def axis_sum(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.sum(arr, axis=axis)

def axis_mean(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(arr, axis=axis)

def axis_min(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.min(arr, axis=axis)

def axis_max(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.max(arr, axis=axis)

def axis_std(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.std(arr, axis=axis)

def axis_var(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.var(arr, axis=axis)
