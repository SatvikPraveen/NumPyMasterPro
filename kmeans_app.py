import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.kmeans_utils import kmeans, initialize_centroids, assign_clusters, update_centroids

st.title("🧠 K-Means Clustering Explorer")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to cluster.")
    else:
        x_col = st.selectbox("X-axis feature", numeric_cols)
        y_col = st.selectbox("Y-axis feature", numeric_cols, index=1)

        k = st.slider("Number of clusters (k)", 1, 10, value=3)
        run = st.button("Run K-Means")

        if run:
            data = df[[x_col, y_col]].dropna().values
            centroids, labels = kmeans(data, k)

            for i in range(k):
                cluster_data = data[labels == i]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i+1}")

            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X', label='Centroids')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title("KMeans Clustering Result")
            plt.legend()
            st.pyplot(plt)
