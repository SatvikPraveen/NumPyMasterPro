"""
Unit tests for math_utils module
"""
import pytest
import numpy as np
from scripts.math_utils import (
    add_arrays,
    subtract_arrays,
    multiply_arrays,
    divide_arrays,
    power_array,
    sqrt_array,
    exp_array,
    natural_log,
    sin_array,
    cos_array,
    round_array,
    floor_array,
    ceil_array,
    clip_array,
    absolute_array,
    cumulative_sum,
    cumulative_product,
)


class TestBasicArithmetic:
    """Tests for basic arithmetic operations"""
    
    def test_add_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = add_arrays(arr1, arr2)
        expected = np.array([5, 7, 9])
        assert np.array_equal(result, expected)
    
    def test_subtract_arrays(self):
        arr1 = np.array([10, 20, 30])
        arr2 = np.array([1, 2, 3])
        result = subtract_arrays(arr1, arr2)
        expected = np.array([9, 18, 27])
        assert np.array_equal(result, expected)
    
    def test_multiply_arrays(self):
        arr1 = np.array([2, 3, 4])
        arr2 = np.array([5, 6, 7])
        result = multiply_arrays(arr1, arr2)
        expected = np.array([10, 18, 28])
        assert np.array_equal(result, expected)
    
    def test_divide_arrays(self):
        arr1 = np.array([10, 20, 30])
        arr2 = np.array([2, 4, 5])
        result = divide_arrays(arr1, arr2)
        expected = np.array([5, 5, 6])
        assert np.array_equal(result, expected)


class TestPowerAndRoots:
    """Tests for power and root operations"""
    
    def test_power_array(self):
        arr = np.array([2, 3, 4])
        result = power_array(arr, 2)
        expected = np.array([4, 9, 16])
        assert np.array_equal(result, expected)
    
    def test_sqrt_array(self):
        arr = np.array([4, 9, 16])
        result = sqrt_array(arr)
        expected = np.array([2, 3, 4])
        assert np.array_equal(result, expected)
    
    def test_sqrt_negative_warning(self):
        # sqrt of negative should produce warning/nan
        arr = np.array([-4, 9])
        result = sqrt_array(arr)
        assert np.isnan(result[0])
        assert result[1] == 3


class TestExponentialAndLog:
    """Tests for exponential and logarithm operations"""
    
    def test_exp_array(self):
        arr = np.array([0, 1, 2])
        result = exp_array(arr)
        expected = np.array([1, np.e, np.e**2])
        assert np.allclose(result, expected)
    
    def test_natural_log(self):
        arr = np.array([1, np.e, np.e**2])
        result = natural_log(arr)
        expected = np.array([0, 1, 2])
        assert np.allclose(result, expected)


class TestTrigonometric:
    """Tests for trigonometric functions"""
    
    def test_sin_array(self):
        arr = np.array([0, np.pi/2, np.pi])
        result = sin_array(arr)
        expected = np.array([0, 1, 0])
        assert np.allclose(result, expected, atol=1e-10)
    
    def test_cos_array(self):
        arr = np.array([0, np.pi/2, np.pi])
        result = cos_array(arr)
        expected = np.array([1, 0, -1])
        assert np.allclose(result, expected, atol=1e-10)


class TestRounding:
    """Tests for rounding operations"""
    
    def test_round_array(self):
        arr = np.array([1.234, 2.567, 3.891])
        result = round_array(arr, decimals=1)
        expected = np.array([1.2, 2.6, 3.9])
        assert np.allclose(result, expected)
    
    def test_floor_array(self):
        arr = np.array([1.9, 2.1, 3.8])
        result = floor_array(arr)
        expected = np.array([1, 2, 3])
        assert np.array_equal(result, expected)
    
    def test_ceil_array(self):
        arr = np.array([1.1, 2.9, 3.2])
        result = ceil_array(arr)
        expected = np.array([2, 3, 4])
        assert np.array_equal(result, expected)


class TestClipAndAbsolute:
    """Tests for clipping and absolute value operations"""
    
    def test_clip_array(self):
        arr = np.array([1, 5, 10, 15, 20])
        result = clip_array(arr, min_val=5, max_val=15)
        expected = np.array([5, 5, 10, 15, 15])
        assert np.array_equal(result, expected)
    
    def test_absolute_array(self):
        arr = np.array([-3, -1, 0, 2, 5])
        result = absolute_array(arr)
        expected = np.array([3, 1, 0, 2, 5])
        assert np.array_equal(result, expected)


class TestCumulative:
    """Tests for cumulative operations"""
    
    def test_cumulative_sum(self):
        arr = np.array([1, 2, 3, 4])
        result = cumulative_sum(arr)
        expected = np.array([1, 3, 6, 10])
        assert np.array_equal(result, expected)
    
    def test_cumulative_product(self):
        arr = np.array([1, 2, 3, 4])
        result = cumulative_product(arr)
        expected = np.array([1, 2, 6, 24])
        assert np.array_equal(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
