import numpy as np

def describe_array(arr):
    print("Shape:", arr.shape)
    print("Size:", arr.size)
    print("Ndim:", arr.ndim)
    print("Dtype:", arr.dtype)
    print("Memory (bytes):", arr.nbytes)

def create_identity_matrix(n):
    return np.eye(n)

def generate_range(start, stop, step=1):
    return np.arange(start, stop, step)