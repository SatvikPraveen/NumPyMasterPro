"""
array_utils.py

Inspect, compare and reason about NumPy arrays: metadata, memory layout,
views vs. copies, and broadcasting.

Author: Satvik Praveen
Project: NumPyMasterPro
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "array_flags",
    "array_summary_table",
    "broadcast_result_shape",
    "compare_arrays",
    "create_identity_matrix",
    "describe_array",
    "explain_broadcast",
    "flatten_or_ravel",
    "generate_range",
    "human_bytes",
    "is_view_of",
    "print_array_with_header",
]

_FLAG_NAMES = ("C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED", "WRITEBACKIFCOPY")


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string (``1536 -> '1.50 KiB'``)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.2f} TiB"  # pragma: no cover - unreachable


def describe_array(arr: npt.ArrayLike, verbose: bool = True) -> dict[str, Any]:
    """
    Key metadata about an array.

    Returns shape, size, ndim, dtype, itemsize, nbytes, strides, whether the
    array owns its memory (``is_view`` is True for views/slices) and a
    human-readable memory figure.

    Parameters
    ----------
    arr : array_like
    verbose : bool
        Print the summary as well as returning it.
    """
    a = np.asarray(arr)
    info: dict[str, Any] = {
        "shape": a.shape,
        "size": a.size,
        "ndim": a.ndim,
        "dtype": a.dtype,
        "itemsize": a.itemsize,
        "nbytes": a.nbytes,
        "strides": a.strides,
        "is_view": a.base is not None,
        "memory": human_bytes(a.nbytes),
    }
    if verbose:
        print("🔍 Array Summary:")
        for k, v in info.items():
            print(f"  {k:<10}: {v}")
    return info


def array_flags(arr: npt.ArrayLike, verbose: bool = True) -> dict[str, bool]:
    """
    Memory-layout flags of an array as a ``{name: bool}`` dict.

    Reads ``ndarray.flags`` attributes directly rather than parsing its string
    representation, so it is robust to formatting changes between NumPy versions.
    """
    a = np.asarray(arr)
    flags = {name: bool(getattr(a.flags, name.lower())) for name in _FLAG_NAMES}
    if verbose:
        print("🧠 Memory Flags:")
        for k, v in flags.items():
            print(f"  - {k}: {v}")
    return flags


def flatten_or_ravel(arr: np.ndarray, use_view: bool = True) -> np.ndarray:
    """
    Flatten with ``ravel`` (a view when possible) or ``flatten`` (always a copy).

    Parameters
    ----------
    arr : ndarray
    use_view : bool
        True → ``ravel()``; False → ``flatten()``.
    """
    return arr.ravel() if use_view else arr.flatten()


def is_view_of(candidate: np.ndarray, base: np.ndarray) -> bool:
    """
    True if ``candidate`` shares memory with ``base`` (i.e. writes propagate).

    Uses ``np.shares_memory``, which is exact (unlike ``may_share_memory``).
    """
    return bool(np.shares_memory(candidate, base))


def compare_arrays(
    arr1: npt.ArrayLike,
    arr2: npt.ArrayLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """
    Compare two arrays on shape, dtype, exact equality and approximate equality.

    ``allclose`` is False when shapes differ, rather than raising.
    """
    a = np.asarray(arr1)
    b = np.asarray(arr2)
    same_shape = a.shape == b.shape
    numeric = np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
    return {
        "shape_equal": same_shape,
        "dtype_equal": a.dtype == b.dtype,
        "elementwise_equal": bool(np.array_equal(a, b)),
        "allclose": bool(same_shape and numeric and np.allclose(a, b, rtol=rtol, atol=atol)),
        "shares_memory": bool(np.shares_memory(a, b)),
    }


def print_array_with_header(arr: npt.ArrayLike, header: str = "") -> None:
    """Print an optional header followed by the array."""
    if header:
        print(f"\n🔹 {header}")
    print(np.asarray(arr))


def array_summary_table(*arrays: npt.ArrayLike, names: Sequence[str] | None = None) -> pd.DataFrame:
    """
    One-row-per-array summary as a pandas DataFrame (pandas is imported lazily).

    Parameters
    ----------
    *arrays : array_like
    names : sequence of str, optional
        Row labels; defaults to ``Array 1``, ``Array 2``, ...
    """
    import pandas as pd

    if names is not None and len(names) != len(arrays):
        raise ValueError("names must have one entry per array")
    rows = []
    for i, arr in enumerate(arrays):
        a = np.asarray(arr)
        rows.append(
            {
                "Name": names[i] if names else f"Array {i + 1}",
                "Shape": a.shape,
                "Size": a.size,
                "Dtype": a.dtype,
                "Nbytes": a.nbytes,
                "Memory": human_bytes(a.nbytes),
                "C_Contiguous": bool(a.flags.c_contiguous),
            }
        )
    return pd.DataFrame(rows)


def create_identity_matrix(n: int) -> npt.NDArray[np.float64]:
    """Identity matrix of shape ``(n, n)``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.eye(n)


def generate_range(start: float, stop: float, step: float = 1) -> np.ndarray:
    """``np.arange(start, stop, step)`` with a guard against a zero step."""
    if step == 0:
        raise ValueError("step must be non-zero")
    return np.arange(start, stop, step)


# ---------------------------------------------------------------------------
# Broadcasting helpers
# ---------------------------------------------------------------------------
def broadcast_result_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    """
    Result shape of broadcasting the given shapes together, or ``ValueError``.

    Implements the rule by hand (right-align, dims must be equal or 1) so the
    logic is visible; ``np.broadcast_shapes`` gives the same answer.
    """
    ndim = max((len(s) for s in shapes), default=0)
    result: list[int] = []
    for axis in range(1, ndim + 1):
        dims = {s[-axis] for s in shapes if len(s) >= axis}
        non_unit = dims - {1}
        if len(non_unit) > 1:
            raise ValueError(
                f"operands could not be broadcast together with shapes {list(map(tuple, shapes))}"
            )
        result.append(next(iter(non_unit)) if non_unit else 1)
    return tuple(reversed(result))


def explain_broadcast(shape_a: Sequence[int], shape_b: Sequence[int]) -> str:
    """
    Human-readable, axis-by-axis explanation of how two shapes broadcast.

    Returns a multi-line string; the last line reports the result shape or the
    reason broadcasting fails.
    """
    a, b = tuple(shape_a), tuple(shape_b)
    ndim = max(len(a), len(b))
    pa = (1,) * (ndim - len(a)) + a
    pb = (1,) * (ndim - len(b)) + b
    lines = [f"A: {a} -> padded {pa}", f"B: {b} -> padded {pb}"]
    for i, (da, db) in enumerate(zip(pa, pb, strict=True)):
        if da == db:
            verdict = f"equal -> {da}"
        elif da == 1 or db == 1:
            verdict = f"one is 1 -> stretch to {max(da, db)}"
        else:
            verdict = "MISMATCH"
        lines.append(f"axis {i}: {da} vs {db}: {verdict}")
    try:
        lines.append(f"result: {broadcast_result_shape(a, b)}")
    except ValueError as err:
        lines.append(f"result: incompatible ({err})")
    return "\n".join(lines)
