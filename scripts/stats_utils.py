import numpy as np

def summarize(arr):
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "var": np.var(arr),
        "min": np.min(arr),
        "max": np.max(arr)
    }

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def zscore(arr):
    return (arr - np.mean(arr)) / np.std(arr)