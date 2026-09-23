"""
perf_utils.py

Tools for measuring and improving NumPy performance: timing helpers,
vectorised-vs-loop comparisons, stride tricks and chunked processing for
arrays that should not be materialised at once.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided, sliding_window_view

__all__ = [
    "Timer",
    "TimingResult",
    "chunked_apply",
    "iter_chunks",
    "memory_footprint",
    "rolling_windows",
    "strided_blocks",
    "timeit_compare",
]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """
    Context manager measuring wall-clock time with ``perf_counter``.

    Examples
    --------
    >>> with Timer() as t:
    ...     _ = np.arange(10)
    >>> t.elapsed >= 0
    True
    """

    def __init__(self, label: str = "", verbose: bool = False) -> None:
        self.label = label
        self.verbose = verbose
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            prefix = f"{self.label}: " if self.label else ""
            print(f"⏱️  {prefix}{self.elapsed * 1e3:.3f} ms")


@dataclass(frozen=True)
class TimingResult:
    """Per-callable timing statistics from :func:`timeit_compare`."""

    name: str
    best: float  # seconds per call (minimum over repeats)
    mean: float
    std: float
    runs: int

    @property
    def best_ms(self) -> float:
        return self.best * 1e3


def timeit_compare(
    funcs: Mapping[str, Callable[..., Any]],
    *args: Any,
    number: int = 10,
    repeat: int = 5,
    check_equal: bool = True,
    **kwargs: Any,
) -> dict[str, TimingResult]:
    """
    Time several implementations of the same computation on identical inputs.

    Each callable is invoked ``number`` times per repeat, ``repeat`` times; the
    reported ``best`` is the minimum per-call time (the least noisy estimate).
    With ``check_equal`` the outputs are asserted equal with ``np.allclose`` so
    you never compare a fast-but-wrong version.

    Parameters
    ----------
    funcs : mapping of name -> callable
    *args, **kwargs
        Forwarded to every callable.

    Returns
    -------
    dict of name -> TimingResult, in the order given.
    """
    if not funcs:
        raise ValueError("funcs must contain at least one callable")
    if number < 1 or repeat < 1:
        raise ValueError("number and repeat must be >= 1")

    outputs: dict[str, Any] = {}
    results: dict[str, TimingResult] = {}
    for name, fn in funcs.items():
        outputs[name] = fn(*args, **kwargs)  # warm-up + capture result
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            for _ in range(number):
                fn(*args, **kwargs)
            samples.append((time.perf_counter() - start) / number)
        arr = np.asarray(samples)
        results[name] = TimingResult(
            name, float(arr.min()), float(arr.mean()), float(arr.std()), repeat
        )

    if check_equal and len(outputs) > 1:
        names = list(outputs)
        reference = np.asarray(outputs[names[0]])
        for other in names[1:]:
            candidate = np.asarray(outputs[other])
            if reference.shape != candidate.shape or not np.allclose(
                reference, candidate, equal_nan=True
            ):
                raise AssertionError(f"Outputs of {names[0]!r} and {other!r} differ")
    return results


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def memory_footprint(arr: np.ndarray) -> dict[str, Any]:
    """
    Bytes owned by an array versus bytes it merely views.

    ``owned`` is the buffer size when the array owns its data; for views it is
    0 and ``base_nbytes`` reports the size of the underlying buffer.
    """
    base = arr.base
    return {
        "nbytes": int(arr.nbytes),
        "owns_data": bool(arr.flags.owndata),
        "owned": int(arr.nbytes) if arr.flags.owndata else 0,
        "base_nbytes": int(getattr(base, "nbytes", 0)) if base is not None else 0,
        "contiguous": bool(arr.flags.c_contiguous or arr.flags.f_contiguous),
    }


# ---------------------------------------------------------------------------
# Stride tricks
# ---------------------------------------------------------------------------
def rolling_windows(arr: npt.ArrayLike, window: int, step: int = 1) -> np.ndarray:
    """
    Overlapping windows of a 1-D array as a zero-copy ``(n_windows, window)`` view.

    Built on ``sliding_window_view``; ``step`` subsamples the windows. The
    result is read-only because the rows alias each other.
    """
    x = np.asarray(arr)
    if x.ndim != 1:
        raise ValueError("rolling_windows expects a 1-D array")
    if not 1 <= window <= x.size:
        raise ValueError(f"window must be in [1, {x.size}]")
    if step < 1:
        raise ValueError("step must be >= 1")
    return sliding_window_view(x, window)[::step]


def strided_blocks(arr: npt.ArrayLike, block: tuple[int, int]) -> np.ndarray:
    """
    Non-overlapping ``block``-sized tiles of a 2-D array as a 4-D view.

    Output shape is ``(rows // bh, cols // bw, bh, bw)``; e.g. block means of
    an image are then ``strided_blocks(img, (2, 2)).mean(axis=(2, 3))``.
    The array must be C-contiguous and its dimensions divisible by the block.
    """
    x = np.ascontiguousarray(arr)
    if x.ndim != 2:
        raise ValueError("strided_blocks expects a 2-D array")
    bh, bw = block
    rows, cols = x.shape
    if bh < 1 or bw < 1 or rows % bh or cols % bw:
        raise ValueError(f"block {block} must be positive and divide shape {x.shape}")
    rs, cs = x.strides
    shape = (rows // bh, cols // bw, bh, bw)
    strides = (rs * bh, cs * bw, rs, cs)
    return as_strided(x, shape=shape, strides=strides, writeable=False)


# ---------------------------------------------------------------------------
# Chunked processing
# ---------------------------------------------------------------------------
def iter_chunks(arr: np.ndarray, chunk_size: int, axis: int = 0) -> Iterator[np.ndarray]:
    """Yield successive slices of length ``chunk_size`` along ``axis`` (views, no copies)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    n = arr.shape[axis]
    for start in range(0, n, chunk_size):
        index: list[slice] = [slice(None)] * arr.ndim
        index[axis] = slice(start, min(start + chunk_size, n))
        yield arr[tuple(index)]


def chunked_apply(
    func: Callable[[np.ndarray], npt.ArrayLike],
    arr: np.ndarray,
    chunk_size: int,
    axis: int = 0,
) -> np.ndarray:
    """
    Apply ``func`` to chunks along ``axis`` and concatenate the results.

    Useful when an operation creates large temporaries (e.g. pairwise
    distances) and the full-array version would exhaust memory. ``func`` must
    be row-separable: applying it to a chunk must equal slicing the full result.
    """
    parts = [np.asarray(func(chunk)) for chunk in iter_chunks(arr, chunk_size, axis)]
    if not parts:
        return np.asarray(func(arr))
    return np.concatenate(parts, axis=axis)
