"""
Unit tests for array_utils module
"""
import pytest
import numpy as np
import pandas as pd
from scripts.array_utils import (
    describe_array,
    array_flags,
    flatten_or_ravel,
    compare_arrays,
    create_identity_matrix,
    generate_range,
)


class TestDescribeArray:
    """Tests for describe_array function"""
    
    def test_1d_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = describe_array(arr, verbose=False)
        
        assert result['shape'] == (5,)
        assert result['size'] == 5
        assert result['ndim'] == 1
        assert result['dtype'] == arr.dtype
    
    def test_2d_array(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        result = describe_array(arr, verbose=False)
        
        assert result['shape'] == (3, 2)
        assert result['size'] == 6
        assert result['ndim'] == 2
    
    def test_itemsize_calculation(self):
        arr_int32 = np.array([1, 2, 3], dtype=np.int32)
        result = describe_array(arr_int32, verbose=False)
        
        assert result['itemsize'] == 4  # int32 is 4 bytes
        assert result['nbytes'] == 12  # 3 elements * 4 bytes


class TestArrayFlags:
    """Tests for array_flags function"""
    
    def test_c_contiguous(self):
        arr = np.array([[1, 2], [3, 4]])  # C-contiguous by default
        flags = array_flags(arr, verbose=False)
        
        assert flags['C_CONTIGUOUS'] is True
    
    def test_fortran_contiguous(self):
        arr = np.array([[1, 2], [3, 4]], order='F')
        flags = array_flags(arr, verbose=False)
        
        assert flags['F_CONTIGUOUS'] is True
    
    def test_writeable(self):
        arr = np.array([1, 2, 3])
        flags = array_flags(arr, verbose=False)
        
        assert flags['WRITEABLE'] is True


class TestFlattenOrRavel:
    """Tests for flatten_or_ravel function"""
    
    def test_flatten_creates_copy(self):
        arr = np.array([[1, 2], [3, 4]])
        result = flatten_or_ravel(arr, use_view=False)
        
        assert result.shape == (4,)
        # Modifying result should not affect original
        result[0] = 999
        assert arr[0, 0] == 1
    
    def test_ravel_creates_view(self):
        arr = np.array([[1, 2], [3, 4]])
        result = flatten_or_ravel(arr, use_view=True)
        
        assert result.shape == (4,)
        # Modifying result should affect original
        result[0] = 999
        assert arr[0, 0] == 999


class TestCompareArrays:
    """Tests for compare_arrays function"""
    
    def test_identical_arrays(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        result = compare_arrays(arr1, arr2)
        
        assert result['shape_equal'] is True
        assert result['dtype_equal'] is True
        assert result['elementwise_equal'] is True
    
    def test_different_shapes(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2])
        result = compare_arrays(arr1, arr2)
        
        assert result['shape_equal'] is False
        assert result['elementwise_equal'] is False
    
    def test_different_values(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        result = compare_arrays(arr1, arr2)
        
        assert result['shape_equal'] is True
        assert result['elementwise_equal'] is False


class TestCreateIdentityMatrix:
    """Tests for create_identity_matrix function"""
    
    def test_3x3_identity(self):
        result = create_identity_matrix(3)
        expected = np.eye(3)
        
        assert np.array_equal(result, expected)
    
    def test_5x5_identity(self):
        result = create_identity_matrix(5)
        
        assert result.shape == (5, 5)
        assert np.allclose(np.sum(result), 5.0)  # Sum of diagonal
        assert np.allclose(result[0, 0], 1.0)
        assert np.allclose(result[0, 1], 0.0)


class TestGenerateRange:
    """Tests for generate_range function"""
    
    def test_basic_range(self):
        result = generate_range(0, 10, 2)
        expected = np.arange(0, 10, 2)
        
        assert np.array_equal(result, expected)
    
    def test_negative_step(self):
        result = generate_range(10, 0, -2)
        expected = np.arange(10, 0, -2)
        
        assert np.array_equal(result, expected)
    
    def test_float_range(self):
        result = generate_range(0.0, 1.0, 0.1)
        
        assert len(result) == 10
        assert np.allclose(result[0], 0.0)
        assert np.allclose(result[-1], 0.9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
