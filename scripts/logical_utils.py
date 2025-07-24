import numpy as np

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
