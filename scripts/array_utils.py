import numpy as np
import pandas as pd

def describe_array(arr: np.ndarray, verbose: bool = True) -> dict:
    """
    Print and return key metadata about a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Array to inspect.
    verbose : bool, optional
        If True, print the metadata. Default is True.

    Returns
    -------
    dict
        Dictionary with array shape, size, ndim, dtype, itemsize, and memory usage.
    """
    info = {
        "shape": arr.shape,
        "size": arr.size,
        "ndim": arr.ndim,
        "dtype": arr.dtype,
        "itemsize": arr.itemsize,
        "nbytes": arr.nbytes
    }
    if verbose:
        print("🔍 Array Summary:")
        for k, v in info.items():
            print(f"  {k.capitalize():<10}: {v}")
    return info


def array_flags(arr: np.ndarray, verbose: bool = True) -> dict:
    """
    Return internal memory layout flags of a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array.
    verbose : bool, optional
        If True, print the flags. Default is True.

    Returns
    -------
    dict
        Dictionary of memory layout flags.
    """
    # .flags is a numpy.flagsobj object, which can be converted to string and parsed
    flag_lines = str(arr.flags).strip().split('\n')
    flags = {}
    for line in flag_lines:
        if ':' in line:
            key, val = line.strip().split(':')
            flags[key.strip()] = val.strip() == 'True'

    if verbose:
        print("🧠 Memory Flags:")
        for k, v in flags.items():
            print(f"  - {k}: {v}")

    return flags


def flatten_or_ravel(arr: np.ndarray, use_view: bool = True) -> np.ndarray:
    """
    Flatten an array using either ravel (view) or flatten (copy).

    Parameters
    ----------
    arr : np.ndarray
    use_view : bool, optional
        If True, use ravel() (view). If False, use flatten() (copy). Default is True.

    Returns
    -------
    np.ndarray
        Flattened array.
    """
    return arr.ravel() if use_view else arr.flatten()


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray) -> dict:
    """
    Compare two arrays in terms of shape, dtype, and elementwise equality.

    Parameters
    ----------
    arr1 : np.ndarray
    arr2 : np.ndarray

    Returns
    -------
    dict
        Comparison results.
    """
    return {
        "shape_equal": arr1.shape == arr2.shape,
        "dtype_equal": arr1.dtype == arr2.dtype,
        "elementwise_equal": np.array_equal(arr1, arr2)
    }


def print_array_with_header(arr: np.ndarray, header: str = "") -> None:
    """
    Print a header followed by the array.

    Parameters
    ----------
    arr : np.ndarray
    header : str, optional
        Title/header to display. Default is "".
    """
    if header:
        print(f"\n🔹 {header}")
    print(arr)


def array_summary_table(*arrays: np.ndarray, names=None) -> pd.DataFrame:
    """
    Create a summary table for one or more arrays.

    Parameters
    ----------
    arrays : np.ndarray
        One or more NumPy arrays.
    names : list, optional
        List of names for the arrays. If None, generic names will be used.

    Returns
    -------
    pd.DataFrame
        Summary table with shape, size, dtype, and memory.
    """
    rows = []
    for i, arr in enumerate(arrays):
        name = names[i] if names else f"Array {i+1}"
        rows.append({
            "Name": name,
            "Shape": arr.shape,
            "Size": arr.size,
            "Dtype": arr.dtype,
            "Nbytes": arr.nbytes
        })
    return pd.DataFrame(rows)


def create_identity_matrix(n: int) -> np.ndarray:
    """
    Create an identity matrix of shape (n, n).

    Parameters
    ----------
    n : int
        Size of the identity matrix.

    Returns
    -------
    np.ndarray
    """
    return np.eye(n)


def generate_range(start: int, stop: int, step: int = 1) -> np.ndarray:
    """
    Generate a range of values as a NumPy array.

    Parameters
    ----------
    start : int
    stop : int
    step : int, optional
        Step size. Default is 1.

    Returns
    -------
    np.ndarray
    """
    return np.arange(start, stop, step)
