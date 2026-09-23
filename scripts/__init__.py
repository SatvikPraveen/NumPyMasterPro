"""
NumPyMasterPro Utility Scripts

This package provides modular utility functions for NumPy operations,
organized by topic for easy reuse across notebooks and projects.

Quick imports:
--------------
from scripts import kmeans, describe_array, minmax_normalize
from scripts.math_utils import power_array, sqrt_array
from scripts.linear_algebra_utils import compute_inverse, eigen_decomposition
"""

# Array Utilities
# Aggregation Utilities
from .aggregation_utils import (
    array_max,
    array_mean,
    array_min,
    array_std,
    array_sum,
    array_var,
    axis_mean,
    axis_sum,
)
from .array_utils import (
    array_flags,
    array_summary_table,
    compare_arrays,
    create_identity_matrix,
    describe_array,
)

# I/O Utilities
from .io_utils import (
    create_memmap,
    load_memmap,
    load_npy,
    load_npz,
    load_txt,
    save_npy,
    save_npz,
    save_txt,
)

# K-Means Utilities
from .kmeans_utils import (
    assign_clusters,
    compute_cluster_inertia,
    compute_inertia,
    generate_data,
    initialize_centroids,
    kmeans,
    update_centroids,
)

# Linear Algebra Utilities
from .linear_algebra_utils import (
    closed_form_linear_regression,
    compute_determinant,
    compute_inverse,
    compute_svd,
    dot_product,
    eigen_decomposition,
    l1_norm,
    l2_norm,
    matmul_product,
    solve_system,
)

# Logical Utilities
from .logical_utils import (
    all_condition,
    any_condition,
    check_finite,
    check_inf,
    check_isin,
    check_nan,
    classify_scores,
    compound_condition,
    count_matching,
    find_indices,
    mask_by_value,
    where_condition,
)

# Math Utilities
from .math_utils import (
    add_arrays,
    clip_array,
    cos_array,
    exp_array,
    multiply_arrays,
    natural_log,
    power_array,
    round_array,
    sin_array,
    sqrt_array,
)

# Statistics Utilities
from .stats_utils import (
    compute_correlation,
    generate_normal_distribution,
    generate_random_integers,
    histogram_binning,
    minmax_normalize,
    summarize_array,
    zscore_normalize,
)

__version__ = "1.0.0"

__all__ = [
    # Math
    "add_arrays",
    "all_condition",
    # Logical
    "any_condition",
    "array_flags",
    "array_max",
    "array_mean",
    "array_min",
    "array_std",
    # Aggregation
    "array_sum",
    "array_summary_table",
    "array_var",
    "assign_clusters",
    "axis_mean",
    "axis_sum",
    "check_finite",
    "check_inf",
    "check_isin",
    "check_nan",
    "classify_scores",
    "clip_array",
    "closed_form_linear_regression",
    "compare_arrays",
    "compound_condition",
    "compute_cluster_inertia",
    "compute_correlation",
    "compute_determinant",
    "compute_inertia",
    "compute_inverse",
    "compute_svd",
    "cos_array",
    "count_matching",
    "create_identity_matrix",
    "create_memmap",
    # Array utilities
    "describe_array",
    # Linear algebra
    "dot_product",
    "eigen_decomposition",
    "exp_array",
    "find_indices",
    "generate_data",
    "generate_normal_distribution",
    "generate_random_integers",
    "histogram_binning",
    "initialize_centroids",
    # K-Means
    "kmeans",
    "l1_norm",
    "l2_norm",
    "load_memmap",
    "load_npy",
    "load_npz",
    "load_txt",
    "mask_by_value",
    "matmul_product",
    "minmax_normalize",
    "multiply_arrays",
    "natural_log",
    "power_array",
    "round_array",
    # I/O
    "save_npy",
    "save_npz",
    "save_txt",
    "sin_array",
    "solve_system",
    "sqrt_array",
    # Statistics
    "summarize_array",
    "update_centroids",
    "where_condition",
    "zscore_normalize",
]
