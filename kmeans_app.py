"""
K-Means Clustering Explorer — Streamlit front-end for ``scripts.kmeans_utils``.

Run with::

    streamlit run kmeans_app.py

Everything numerical is NumPy only: k-means++ seeding, Lloyd iterations,
silhouette scoring, the elbow heuristic and the PCA projection all come from
the ``scripts`` package in this repository.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.kmeans_utils import (
    KMeansResult,
    compute_inertia,
    elbow_point,
    generate_data,
    run_kmeans,
    silhouette_score,
)
from scripts.linear_algebra_utils import pca
from scripts.stats_utils import zscore_normalize

DEMO_CENTERS = ((2.0, 2.0), (7.0, 7.0), (2.0, 7.0), (7.0, 2.0))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_demo(n_per_cluster: int, n_blobs: int, std: float, seed: int) -> pd.DataFrame:
    X = generate_data(
        n_per_cluster=n_per_cluster, centers=DEMO_CENTERS[:n_blobs], std=std, seed=seed
    )
    return pd.DataFrame(X, columns=["x", "y"])


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def prepare_matrix(df: pd.DataFrame, columns: list[str], standardize: bool) -> np.ndarray:
    X = df[columns].dropna().to_numpy(dtype=float)
    return zscore_normalize(X, axis=0) if standardize else X


# ---------------------------------------------------------------------------
# Clustering (cached so slider tweaks that don't change inputs are free)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fit(X: np.ndarray, k: int, init: str, n_init: int, max_iters: int, seed: int) -> KMeansResult:
    return run_kmeans(X, k=k, init=init, n_init=n_init, max_iters=max_iters, seed=seed)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def elbow_curve(X: np.ndarray, k_max: int, init: str, seed: int) -> tuple[list[int], list[float]]:
    ks = list(range(1, k_max + 1))
    return ks, compute_inertia(X, ks, n_init=3, seed=seed, init=init)


def to_2d(X: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Project to 2-D for plotting; PCA when there are more than two features."""
    if X.shape[1] == 2:
        return X, "feature 1", "feature 2"
    res = pca(X, n_components=2)
    r = res.explained_variance_ratio
    return res.transformed, f"PC1 ({r[0]:.0%} var)", f"PC2 ({r[1]:.0%} var)"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def cluster_figure(
    X2: np.ndarray, labels: np.ndarray, centroids2: np.ndarray, xlabel: str, ylabel: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    k = centroids2.shape[0]
    cmap = plt.get_cmap("tab10")
    for i in range(k):
        pts = X2[labels == i]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            alpha=0.75,
            color=cmap(i % 10),
            label=f"Cluster {i + 1} (n={len(pts)})",
        )
    ax.scatter(
        centroids2[:, 0],
        centroids2[:, 1],
        c="black",
        s=220,
        marker="X",
        label="Centroids",
        zorder=5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("K-Means result")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def elbow_figure(ks: list[int], inertias: list[float], suggested: int, chosen: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ks, inertias, marker="o")
    ax.axvline(suggested, color="tab:green", ls="--", label=f"elbow heuristic: k={suggested}")
    ax.axvline(chosen, color="tab:red", ls=":", label=f"current: k={chosen}")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (WCSS)")
    ax.set_title("Elbow curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def convergence_figure(history: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(history) + 1), history, marker=".")
    ax.set_xlabel("Lloyd iteration")
    ax.set_ylabel("inertia")
    ax.set_title("Convergence of the best run")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="K-Means Explorer", page_icon="🧠", layout="wide")
    st.title("🧠 K-Means Clustering Explorer")
    st.caption("NumPy-only k-means++ with restarts, silhouette scoring and an elbow heuristic.")

    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Demo blobs", "Upload CSV"], horizontal=True)
        if source == "Demo blobs":
            n_blobs = st.slider("Number of blobs", 2, 4, 3)
            n_per = st.slider("Points per blob", 20, 300, 60, step=10)
            std = st.slider("Blob spread (std)", 0.2, 2.5, 1.0, step=0.1)
            data_seed = st.number_input("Data seed", 0, 9999, 42)
            df = load_demo(n_per, n_blobs, std, int(data_seed))
        else:
            uploaded = st.file_uploader("CSV file", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV with at least two numeric columns.")
                st.stop()
            df = pd.read_csv(uploaded)

        cols = numeric_columns(df)
        if len(cols) < 2:
            st.error("Need at least 2 numeric columns to cluster.")
            st.stop()
        features = st.multiselect("Features", cols, default=cols[: min(len(cols), 4)])
        if len(features) < 2:
            st.warning("Select at least two features.")
            st.stop()
        standardize = st.checkbox("Standardise features (z-score)", value=len(features) > 2)

        st.header("Algorithm")
        k = st.slider("Clusters (k)", 1, 10, 3)
        init = st.selectbox("Initialisation", ["k-means++", "random"])
        n_init = st.slider("Restarts (n_init)", 1, 20, 5)
        max_iters = st.slider("Max iterations", 10, 500, 100, step=10)
        seed = st.number_input("Algorithm seed", 0, 9999, 0)
        show_elbow = st.checkbox("Show elbow curve", value=True)

    X = prepare_matrix(df, features, standardize)
    if X.shape[0] < k:
        st.error(f"Only {X.shape[0]} complete rows; choose k ≤ {X.shape[0]}.")
        st.stop()

    left, right = st.columns([3, 2])
    with right:
        st.subheader("Data preview")
        st.dataframe(df[features].head(10))
        st.write(
            f"{X.shape[0]} rows × {X.shape[1]} features"
            + (" (standardised)" if standardize else "")
        )

    result = fit(X, k, init, n_init, max_iters, int(seed))
    X2, xl, yl = to_2d(X)
    if X.shape[1] == 2:
        centroids2 = result.centroids
    else:
        # Project centroids with the same PCA basis as the data.
        res = pca(X, n_components=2)
        centroids2 = (result.centroids - res.mean) @ res.components.T

    with left:
        st.subheader("Clusters")
        st.pyplot(cluster_figure(X2, result.labels, centroids2, xl, yl), clear_figure=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Inertia (WCSS)", f"{result.inertia:,.2f}")
    m2.metric("Iterations", result.n_iter, help="Lloyd iterations of the best restart")
    m3.metric("Converged", "yes" if result.converged else "no")
    if 1 < k < X.shape[0]:
        m4.metric(
            "Silhouette",
            f"{silhouette_score(X, result.labels):.3f}",
            help="1 = well separated, 0 = overlapping",
        )
    else:
        m4.metric("Silhouette", "n/a")

    with st.expander("Convergence of the best run"):
        st.pyplot(convergence_figure(result.inertia_history), clear_figure=True)

    if show_elbow:
        st.subheader("Choosing k")
        k_max = min(10, X.shape[0] - 1)
        ks, inertias = elbow_curve(X, k_max, init, int(seed))
        suggested = elbow_point(ks, inertias)
        st.pyplot(elbow_figure(ks, inertias, suggested, k), clear_figure=True)
        st.caption(f"Elbow heuristic (max distance to chord) suggests **k = {suggested}**.")

    st.subheader("Export")
    labelled = df.loc[df[features].dropna().index].copy()
    labelled["cluster"] = result.labels
    buf = io.StringIO()
    labelled.to_csv(buf, index=False)
    st.download_button(
        "Download labelled CSV", buf.getvalue(), file_name="kmeans_labels.csv", mime="text/csv"
    )

    with st.expander("Centroids"):
        cent = pd.DataFrame(result.centroids, columns=features)
        cent.index = [f"Cluster {i + 1}" for i in range(k)]
        st.dataframe(cent)


if __name__ == "__main__":
    main()
