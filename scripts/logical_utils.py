import numpy as np
from typing import Any, Union

def filter_valid_positive(arr: np.ndarray) -> np.ndarray:
    """Returns values that are finite and greater than 0."""
    return arr[np.isfinite(arr) & (arr > 0)]

def classify_scores(arr: np.ndarray) -> np.ndarray:
    """Classifies scores using vectorized logic."""
    return np.select(
        [arr >= 85, arr >= 60],
        ["Excellent", "Average"],
        default="Fail"
    )

def any_condition(arr: np.ndarray, condition: np.ndarray = None, axis: int = None) -> Union[bool, np.ndarray]:
    """
    Test whether any array element along a given axis evaluates to True.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    condition : np.ndarray, optional
        Boolean mask to test. If None, tests arr directly
    axis : int, optional
        Axis along which to perform the operation
    
    Returns:
    --------
    bool or np.ndarray
        True if any element is True, otherwise False
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> any_condition(arr, arr > 3)
    True
    """
    if condition is not None:
        return np.any(condition, axis=axis)
    return np.any(arr, axis=axis)

def all_condition(arr: np.ndarray, condition: np.ndarray = None, axis: int = None) -> Union[bool, np.ndarray]:
    """
    Test whether all array elements along a given axis evaluate to True.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    condition : np.ndarray, optional
        Boolean mask to test. If None, tests arr directly
    axis : int, optional
        Axis along which to perform the operation
    
    Returns:
    --------
    bool or np.ndarray
        True if all elements are True, otherwise False
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> all_condition(arr, arr > 0)
    True
    """
    if condition is not None:
        return np.all(condition, axis=axis)
    return np.all(arr, axis=axis)

def where_condition(condition: np.ndarray, x: Any = None, y: Any = None) -> Union[tuple, np.ndarray]:
    """
    Return elements chosen from x or y depending on condition.
    
    Parameters:
    -----------
    condition : np.ndarray
        Boolean condition array
    x : scalar or array, optional
        Values to use where condition is True
    y : scalar or array, optional
        Values to use where condition is False
    
    Returns:
    --------
    tuple or np.ndarray
        If x and y are provided, returns array with selected values
        Otherwise, returns tuple of indices where condition is True
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> where_condition(arr > 3, arr, 0)
    array([0, 0, 0, 4, 5])
    """
    if x is not None and y is not None:
        return np.where(condition, x, y)
    return np.where(condition)

def mask_by_value(arr: np.ndarray, value: Any, operator: str = '==') -> np.ndarray:
    """
    Create a boolean mask based on a comparison operator.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    value : scalar
        Value to compare against
    operator : str
        Comparison operator: '==', '!=', '>', '<', '>=', '<='
    
    Returns:
    --------
    np.ndarray
        Boolean mask array
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> mask_by_value(arr, 3, '>')
    array([False, False, False,  True,  True])
    """
    operators = {
        '==': arr == value,
        '!=': arr != value,
        '>': arr > value,
        '<': arr < value,
        '>=': arr >= value,
        '<=': arr <= value
    }
    
    if operator not in operators:
        raise ValueError(f"Invalid operator '{operator}'. Must be one of: {list(operators.keys())}")
    
    return operators[operator]

def count_matching(arr: np.ndarray, condition: np.ndarray) -> int:
    """
    Count the number of elements that match a condition.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    condition : np.ndarray
        Boolean condition mask
    
    Returns:
    --------
    int
        Number of True values in the condition
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> count_matching(arr, arr > 3)
    2
    """
    return np.count_nonzero(condition)

def find_indices(arr: np.ndarray, condition: np.ndarray = None) -> tuple:
    """
    Find indices of nonzero elements or elements matching a condition.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    condition : np.ndarray, optional
        Boolean condition mask. If None, finds nonzero elements in arr
    
    Returns:
    --------
    tuple
        Tuple of arrays, one for each dimension, containing indices
        
    Examples:
    ---------
    >>> arr = np.array([1, 0, 3, 0, 5])
    >>> find_indices(arr)
    (array([0, 2, 4]),)
    """
    if condition is not None:
        return np.nonzero(condition)
    return np.nonzero(arr)

def check_nan(arr: np.ndarray) -> np.ndarray:
    """
    Test element-wise for NaN and return result as a boolean array.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates NaN values
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, np.nan, 4])
    >>> check_nan(arr)
    array([False, False,  True, False])
    """
    return np.isnan(arr)

def check_inf(arr: np.ndarray) -> np.ndarray:
    """
    Test element-wise for positive or negative infinity.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates infinite values
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, np.inf, -np.inf, 5])
    >>> check_inf(arr)
    array([False, False,  True,  True, False])
    """
    return np.isinf(arr)

def check_finite(arr: np.ndarray) -> np.ndarray:
    """
    Test element-wise for finiteness (not infinity and not NaN).
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates finite values
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, np.nan, np.inf, 5])
    >>> check_finite(arr)
    array([ True,  True, False, False,  True])
    """
    return np.isfinite(arr)

def check_isin(arr: np.ndarray, test_values: Union[list, np.ndarray]) -> np.ndarray:
    """
    Calculate element-wise whether values are contained in test_values.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    test_values : list or np.ndarray
        Values to test for membership
    
    Returns:
    --------
    np.ndarray
        Boolean array indicating membership in test_values
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> check_isin(arr, [2, 4, 6])
    array([False,  True, False,  True, False])
    """
    return np.isin(arr, test_values)

def logical_and(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Compute element-wise logical AND.
    
    Parameters:
    -----------
    arr1 : np.ndarray
        First input array
    arr2 : np.ndarray
        Second input array
    
    Returns:
    --------
    np.ndarray
        Boolean array with element-wise AND results
    """
    return np.logical_and(arr1, arr2)

def logical_or(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Compute element-wise logical OR.
    
    Parameters:
    -----------
    arr1 : np.ndarray
        First input array
    arr2 : np.ndarray
        Second input array
    
    Returns:
    --------
    np.ndarray
        Boolean array with element-wise OR results
    """
    return np.logical_or(arr1, arr2)

def logical_not(arr: np.ndarray) -> np.ndarray:
    """
    Compute element-wise logical NOT.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    
    Returns:
    --------
    np.ndarray
        Boolean array with element-wise NOT results
    """
    return np.logical_not(arr)

def compound_condition(arr: np.ndarray, min_val: float = None, max_val: float = None, 
                      exclude_nan: bool = True, exclude_inf: bool = True) -> np.ndarray:
    """
    Create a compound boolean mask with multiple conditions.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
    min_val : float, optional
        Minimum value threshold (inclusive)
    max_val : float, optional
        Maximum value threshold (inclusive)
    exclude_nan : bool, default True
        Whether to exclude NaN values
    exclude_inf : bool, default True
        Whether to exclude infinite values
    
    Returns:
    --------
    np.ndarray
        Boolean mask combining all conditions
        
    Examples:
    ---------
    >>> arr = np.array([1, 2, np.nan, 4, np.inf, 6])
    >>> compound_condition(arr, min_val=2, max_val=5)
    array([False,  True, False,  True, False, False])
    """
    mask = np.ones(arr.shape, dtype=bool)
    
    if exclude_nan:
        mask &= ~np.isnan(arr)
    
    if exclude_inf:
        mask &= ~np.isinf(arr)
    
    if min_val is not None:
        mask &= arr >= min_val
    
    if max_val is not None:
        mask &= arr <= max_val
    
    return mask
