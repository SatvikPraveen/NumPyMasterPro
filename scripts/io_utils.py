import numpy as np
import os

def save_npy(path, arr):
    np.save(path, arr)

def load_npy(path):
    return np.load(path)

def save_npz(path, **kwargs):
    np.savez(path, **kwargs)

def load_npz(path):
    return np.load(path)

def save_txt(path, arr, delimiter=","):
    np.savetxt(path, arr, delimiter=delimiter)

def load_txt(path, delimiter=","):
    return np.loadtxt(path, delimiter=delimiter)

def write_csv_with_missing_values(path):
    with open(path, "w") as f:
        f.write("1.0, 2.0, 3.0\n4.0, , 6.0\n7.0, 8.0, 9.0")

def load_genfromtxt(path, delimiter=","):
    return np.genfromtxt(path, delimiter=delimiter)

def create_memmap(path, shape=(3, 3), dtype='float32'):
    mmap = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    mmap[:] = np.random.rand(*shape)
    return mmap

def load_memmap(path, shape=(3, 3), dtype='float32'):
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)