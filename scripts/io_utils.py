"""
io_utils.py

Persisting and loading NumPy arrays: ``.npy``/``.npz`` binaries, delimited
text, and memory-mapped files for datasets larger than RAM.

All functions accept ``str`` or ``pathlib.Path`` and create parent
directories on write.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

PathLike = str | os.PathLike[str]
MemmapMode = Literal["r+", "r", "w+", "c"]
RandomState = int | np.random.Generator | None

__all__ = [
    "create_memmap",
    "load_genfromtxt",
    "load_memmap",
    "load_npy",
    "load_npz",
    "load_txt",
    "save_npy",
    "save_npz",
    "save_txt",
    "write_csv_with_missing_values",
]


def _prepare(path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Binary formats
# ---------------------------------------------------------------------------
def save_npy(path: PathLike, arr: npt.ArrayLike) -> Path:
    """Save a single array in ``.npy`` format. Returns the written path."""
    p = _prepare(path)
    np.save(p, np.asarray(arr))
    return p if p.suffix == ".npy" else p.with_suffix(p.suffix + ".npy")


def load_npy(path: PathLike, mmap_mode: MemmapMode | None = None) -> np.ndarray:
    """
    Load a ``.npy`` file.

    ``mmap_mode="r"`` maps the file instead of reading it into memory, which is
    the right choice for large arrays you only need to slice.
    """
    return np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)


def save_npz(path: PathLike, compressed: bool = False, **arrays: npt.ArrayLike) -> Path:
    """
    Save several named arrays into one ``.npz`` archive.

    Parameters
    ----------
    path : path-like
    compressed : bool
        Use ``np.savez_compressed`` (slower, smaller) instead of ``np.savez``.
    **arrays
        ``name=array`` pairs; the names become keys on load.
    """
    if not arrays:
        raise ValueError("save_npz needs at least one named array")
    p = _prepare(path)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    writer = np.savez_compressed if compressed else np.savez
    writer(p, **payload)  # type: ignore[arg-type]  # numpy stubs mistype **kwds as bool
    return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")


def load_npz(path: PathLike, as_dict: bool = False) -> np.lib.npyio.NpzFile | dict[str, np.ndarray]:
    """
    Load an ``.npz`` archive.

    By default returns the lazy ``NpzFile`` (arrays are read on access and the
    file handle stays open). With ``as_dict=True`` every array is read eagerly
    into a plain dict and the file is closed.
    """
    npz = np.load(Path(path), allow_pickle=False)
    if not as_dict:
        return npz
    with npz:
        return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------------------
# Text formats
# ---------------------------------------------------------------------------
def save_txt(
    path: PathLike, arr: npt.ArrayLike, delimiter: str = ",", fmt: str = "%.18e", header: str = ""
) -> Path:
    """Save a 1-D or 2-D array as delimited text."""
    p = _prepare(path)
    np.savetxt(
        p,
        np.asarray(arr),
        delimiter=delimiter,
        fmt=fmt,
        header=header,
        comments="" if header else "# ",
    )
    return p


def load_txt(path: PathLike, delimiter: str = ",", skiprows: int = 0) -> np.ndarray:
    """Load delimited text with ``np.loadtxt`` (fast, but every field must parse)."""
    return np.loadtxt(Path(path), delimiter=delimiter, skiprows=skiprows)


def write_csv_with_missing_values(path: PathLike) -> Path:
    """Write a tiny 3x3 CSV with one empty field, for demonstrating ``genfromtxt``."""
    p = _prepare(path)
    p.write_text("1.0, 2.0, 3.0\n4.0, , 6.0\n7.0, 8.0, 9.0\n")
    return p


def load_genfromtxt(
    path: PathLike, delimiter: str = ",", filling_values: float | None = None
) -> np.ndarray:
    """
    Load delimited text tolerating missing fields.

    Missing values become ``NaN`` unless ``filling_values`` is given.
    """
    return np.genfromtxt(Path(path), delimiter=delimiter, filling_values=filling_values)


# ---------------------------------------------------------------------------
# Memory maps
# ---------------------------------------------------------------------------
def create_memmap(
    path: PathLike,
    shape: tuple[int, ...] = (3, 3),
    dtype: npt.DTypeLike = "float32",
    fill: npt.ArrayLike | None = None,
    seed: RandomState = None,
) -> np.memmap:
    """
    Create a writable memory-mapped array on disk.

    Parameters
    ----------
    path : path-like
    shape : tuple of int
    dtype : dtype-like
    fill : array_like, optional
        Values to write. Defaults to uniform random numbers from ``seed``.
    seed : int | Generator | None
        Used only when ``fill`` is None.

    Returns
    -------
    np.memmap in ``"w+"`` mode, already flushed to disk.
    """
    p = _prepare(path)
    mmap = np.memmap(p, dtype=dtype, mode="w+", shape=shape)
    if fill is None:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        mmap[:] = rng.random(shape)
    else:
        mmap[:] = np.asarray(fill, dtype=mmap.dtype)
    mmap.flush()
    return mmap


def load_memmap(
    path: PathLike,
    shape: tuple[int, ...] = (3, 3),
    dtype: npt.DTypeLike = "float32",
    mode: MemmapMode = "r",
) -> np.memmap:
    """Open an existing raw memory-mapped file (read-only by default)."""
    return np.memmap(Path(path), dtype=dtype, mode=mode, shape=shape)
