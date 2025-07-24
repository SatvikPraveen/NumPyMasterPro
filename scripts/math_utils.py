"""
math_utils.py

Reusable utility functions for mathematical and element-wise operations in NumPy.
Includes arithmetic, rounding, exponentials, logarithms, roots, powers,
trigonometry, and clipping operations.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

import numpy as np

# ✅ Element-wise arithmetic
def add_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.add(a, b)

def subtract_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.subtract(a, b)

def multiply_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.multiply(a, b)

def divide_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b)

def modulo_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mod(a, b)

def floor_divide_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.floor_divide(a, b)

def power_array(a: np.ndarray, exponent: float) -> np.ndarray:
    return np.power(a, exponent)


# ✅ Rounding operations
def round_array(arr: np.ndarray, decimals: int = 0) -> np.ndarray:
    return np.round(arr, decimals)

def floor_array(arr: np.ndarray) -> np.ndarray:
    return np.floor(arr)

def ceil_array(arr: np.ndarray) -> np.ndarray:
    return np.ceil(arr)

def trunc_array(arr: np.ndarray) -> np.ndarray:
    return np.trunc(arr)


# ✅ Exponentials and logarithms
def exp_array(arr: np.ndarray) -> np.ndarray:
    return np.exp(arr)

def natural_log(arr: np.ndarray) -> np.ndarray:
    return np.log(arr)

def base10_log(arr: np.ndarray) -> np.ndarray:
    return np.log10(arr)


# ✅ Roots and powers
def sqrt_array(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr)

def cbrt_array(arr: np.ndarray) -> np.ndarray:
    return np.cbrt(arr)


# ✅ Trigonometry
def sin_array(arr: np.ndarray) -> np.ndarray:
    return np.sin(arr)

def cos_array(arr: np.ndarray) -> np.ndarray:
    return np.cos(arr)

def arcsin_array(arr: np.ndarray) -> np.ndarray:
    return np.arcsin(arr)

def to_degrees(arr: np.ndarray) -> np.ndarray:
    return np.degrees(arr)

def to_radians(arr: np.ndarray) -> np.ndarray:
    return np.radians(arr)


# ✅ Clipping and absolute
def clip_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return np.clip(arr, min_val, max_val)

def absolute_array(arr: np.ndarray) -> np.ndarray:
    return np.abs(arr)


# ✅ Cumulative operations
def cumulative_sum(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr)

def cumulative_product(arr: np.ndarray) -> np.ndarray:
    return np.cumprod(arr)
