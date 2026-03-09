"""
Unit tests for logical_utils module
"""
import pytest
import numpy as np
from scripts.logical_utils import (
    any_condition,
    all_condition,
    where_condition,
    mask_by_value,
    count_matching,
    find_indices,
    check_nan,
    check_inf,
    check_finite,
    check_isin,
    compound_condition,
    classify_scores,
    filter_valid_positive,
)


class TestAnyCondition:
    """Tests for any_condition function"""
    
    def test_any_true(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = any_condition(arr, arr > 3)
        assert result == True
    
    def test_any_false(self):
        arr = np.array([1, 2, 3])
        result = any_condition(arr, arr > 10)
        assert result == False
    
    def test_any_with_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        result = any_condition(arr, arr > 2, axis=0)
        assert np.array_equal(result, np.array([True, True]))


class TestAllCondition:
    """Tests for all_condition function"""
    
    def test_all_true(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = all_condition(arr, arr > 0)
        assert result == True
    
    def test_all_false(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = all_condition(arr, arr > 3)
        assert result == False
    
    def test_all_with_axis(self):
        arr = np.array([[1, 2], [3, 4]])
        result = all_condition(arr, arr > 0, axis=1)
        assert np.array_equal(result, np.array([True, True]))


class TestWhereCondition:
    """Tests for where_condition function"""
    
    def test_where_with_values(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = where_condition(arr > 3, arr, 0)
        expected = np.array([0, 0, 0, 4, 5])
        assert np.array_equal(result, expected)
    
    def test_where_indices_only(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = where_condition(arr > 3)
        assert np.array_equal(result[0], np.array([3, 4]))
    
    def test_where_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = where_condition(arr > 2, 1, 0)
        expected = np.array([[0, 0], [1, 1]])
        assert np.array_equal(result, expected)


class TestMaskByValue:
    """Tests for mask_by_value function"""
    
    def test_equal_operator(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = mask_by_value(arr, 3, '==')
        expected = np.array([False, False, True, False, False])
        assert np.array_equal(result, expected)
    
    def test_greater_than(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = mask_by_value(arr, 3, '>')
        expected = np.array([False, False, False, True, True])
        assert np.array_equal(result, expected)
    
    def test_less_equal(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = mask_by_value(arr, 3, '<=')
        expected = np.array([True, True, True, False, False])
        assert np.array_equal(result, expected)
    
    def test_invalid_operator(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            mask_by_value(arr, 2, 'invalid')


class TestCountMatching:
    """Tests for count_matching function"""
    
    def test_count_matching(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = count_matching(arr, arr > 3)
        assert result == 2
    
    def test_count_all_false(self):
        arr = np.array([1, 2, 3])
        result = count_matching(arr, arr > 10)
        assert result == 0
    
    def test_count_all_true(self):
        arr = np.array([1, 2, 3])
        result = count_matching(arr, arr > 0)
        assert result == 3


class TestFindIndices:
    """Tests for find_indices function"""
    
    def test_nonzero_elements(self):
        arr = np.array([1, 0, 3, 0, 5])
        result = find_indices(arr)
        assert np.array_equal(result[0], np.array([0, 2, 4]))
    
    def test_with_condition(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = find_indices(arr, arr > 3)
        assert np.array_equal(result[0], np.array([3, 4]))
    
    def test_2d_array(self):
        arr = np.array([[0, 1], [2, 0]])
        result = find_indices(arr)
        assert np.array_equal(result[0], np.array([0, 1]))
        assert np.array_equal(result[1], np.array([1, 0]))


class TestCheckNan:
    """Tests for check_nan function"""
    
    def test_with_nan(self):
        arr = np.array([1, 2, np.nan, 4])
        result = check_nan(arr)
        expected = np.array([False, False, True, False])
        assert np.array_equal(result, expected)
    
    def test_without_nan(self):
        arr = np.array([1, 2, 3, 4])
        result = check_nan(arr)
        assert not np.any(result)


class TestCheckInf:
    """Tests for check_inf function"""
    
    def test_with_inf(self):
        arr = np.array([1, 2, np.inf, -np.inf, 5])
        result = check_inf(arr)
        expected = np.array([False, False, True, True, False])
        assert np.array_equal(result, expected)
    
    def test_without_inf(self):
        arr = np.array([1, 2, 3, 4])
        result = check_inf(arr)
        assert not np.any(result)


class TestCheckFinite:
    """Tests for check_finite function"""
    
    def test_mixed_values(self):
        arr = np.array([1, 2, np.nan, np.inf, 5])
        result = check_finite(arr)
        expected = np.array([True, True, False, False, True])
        assert np.array_equal(result, expected)
    
    def test_all_finite(self):
        arr = np.array([1, 2, 3, 4])
        result = check_finite(arr)
        assert np.all(result)


class TestCheckIsin:
    """Tests for check_isin function"""
    
    def test_membership(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = check_isin(arr, [2, 4, 6])
        expected = np.array([False, True, False, True, False])
        assert np.array_equal(result, expected)
    
    def test_no_matches(self):
        arr = np.array([1, 2, 3])
        result = check_isin(arr, [4, 5, 6])
        assert not np.any(result)


class TestCompoundCondition:
    """Tests for compound_condition function"""
    
    def test_min_max_range(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        result = compound_condition(arr, min_val=2, max_val=5)
        expected = np.array([False, True, True, True, True, False])
        assert np.array_equal(result, expected)
    
    def test_exclude_nan(self):
        arr = np.array([1, 2, np.nan, 4, 5])
        result = compound_condition(arr, min_val=2, max_val=5, exclude_nan=True)
        expected = np.array([False, True, False, True, True])
        assert np.array_equal(result, expected)
    
    def test_exclude_inf(self):
        arr = np.array([1, 2, np.inf, 4, 5])
        result = compound_condition(arr, min_val=2, max_val=5, exclude_inf=True)
        expected = np.array([False, True, False, True, True])
        assert np.array_equal(result, expected)


class TestClassifyScores:
    """Tests for classify_scores function"""
    
    def test_score_classification(self):
        arr = np.array([95, 75, 50])
        result = classify_scores(arr)
        expected = np.array(["Excellent", "Average", "Fail"])
        assert np.array_equal(result, expected)


class TestFilterValidPositive:
    """Tests for filter_valid_positive function"""
    
    def test_filter_valid_positive(self):
        arr = np.array([1, -2, np.nan, 4, np.inf, 5])
        result = filter_valid_positive(arr)
        expected = np.array([1, 4, 5])
        assert np.array_equal(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
