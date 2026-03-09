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
from .array_utils import (
    describe_array,
    array_flags,
    compare_arrays,
    array_summary_table,
    create_identity_matrix,
)

# Linear Algebra Utilities
from .linear_algebra_utils import (
    dot_product,
    matmul_product,
    compute_inverse,
    compute_determinant,
    eigen_decomposition,
    compute_svd,
    l2_norm,
    l1_norm,
    solve_system,
    closed_form_linear_regression,
)

# Math Utilities
from .math_utils import (
    add_arrays,
    multiply_arrays,
    power_array,
    sqrt_array,
    exp_array,
    natural_log,
    sin_array,
    cos_array,
    clip_array,
    round_array,
)

# Aggregation Utilities
from .aggregation_utils import (
    array_sum,
    array_mean,
    array_min,
    array_max,
    array_std,
    array_var,
    axis_sum,
    axis_mean,
)

# Statistics Utilities
from .stats_utils import (
    summarize_array,
    minmax_normalize,
    zscore_normalize,
    compute_correlation,
    histogram_binning,
    generate_random_integers,
    generate_normal_distribution,
)

# Logical Utilities
from .logical_utils import (
    any_condition,
    all_condition,
    where_condition,
    mask_by_value,
    count_matching,
    find_indices,
    check_nan,
    check_inf,
    check_finite,
    check_isin,
    compound_condition,
    classify_scores,
)

# K-Means Utilities
from .kmeans_utils import (
    kmeans,
    initialize_centroids,
    assign_clusters,
    update_centroids,
    compute_inertia,
    compute_cluster_inertia,
    generate_data,
)

# I/O Utilities
from .io_utils import (
    save_npy,
    load_npy,
    save_npz,
    load_npz,
    save_txt,
    load_txt,
    create_memmap,
    load_memmap,
)

__version__ = "1.0.0"

__all__ = [
    # Array utilities
    "describe_array",
    "array_flags",
    "compare_arrays",
    "array_summary_table",
    "create_identity_matrix",
    # Linear algebra
    "dot_product",
    "matmul_product",
    "compute_inverse",
    "compute_determinant",
    "eigen_decomposition",
    "compute_svd",
    "l2_norm",
    "l1_norm",
    "solve_system",
    "closed_form_linear_regression",
    # Math
    "add_arrays",
    "multiply_arrays",
    "power_array",
    "sqrt_array",
    "exp_array",
    "natural_log",
    "sin_array",
    "cos_array",
    "clip_array",
    "round_array",
    # Aggregation
    "array_sum",
    "array_mean",
    "array_min",
    "array_max",
    "array_std",
    "array_var",
    "axis_sum",
    "axis_mean",
    # Statistics
    "summarize_array",
    "minmax_normalize",
    "zscore_normalize",
    "compute_correlation",
    "histogram_binning",
    "generate_random_integers",
    "generate_normal_distribution",
    # Logical
    "any_condition",
    "all_condition",
    "where_condition",
    "mask_by_value",
    "count_matching",
    "find_indices",
    "check_nan",
    "check_inf",
    "check_finite",
    "check_isin",
    "compound_condition",
    "classify_scores",
    # K-Means
    "kmeans",
    "initialize_centroids",
    "assign_clusters",
    "update_centroids",
    "compute_inertia",
    "compute_cluster_inertia",
    "generate_data",
    # I/O
    "save_npy",
    "load_npy",
    "save_npz",
    "load_npz",
    "save_txt",
    "load_txt",
    "create_memmap",
    "load_memmap",
]
