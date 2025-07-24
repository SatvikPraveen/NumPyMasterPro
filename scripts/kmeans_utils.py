import numpy as np

def generate_data():
    """
    Generates synthetic 2D data with 3 clusters of 50 points each.
    """
    np.random.seed(42)
    cluster_1 = np.random.randn(50, 2) + [2, 2]
    cluster_2 = np.random.randn(50, 2) + [7, 7]
    cluster_3 = np.random.randn(50, 2) + [2, 7]
    return np.vstack((cluster_1, cluster_2, cluster_3))

def initialize_centroids(X, k):
    """
    Randomly selects `k` unique data points as initial centroids.

    Parameters:
    - X (np.ndarray): Dataset of shape (n_samples, n_features)
    - k (int): Number of clusters to initialize

    Returns:
    - np.ndarray: Initial centroids of shape (k, n_features)

    Raises:
    - ValueError: If k > number of samples in X
    """
    n_samples = len(X)
    if k > n_samples:
        raise ValueError(
            f"❌ Cannot initialize {k} centroids with only {n_samples} samples. "
            f"Please choose k ≤ {n_samples}."
        )

    indices = np.random.choice(n_samples, k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    """
    Assigns each point to the nearest centroid based on Euclidean distance.
    """
    dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(dists, axis=1)

def update_centroids(X, labels, k):
    """
    Updates centroids by computing mean of all points assigned to each cluster.
    """
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k=3, max_iters=100, tol=1e-4):
    """
    K-means algorithm: returns final centroids and labels.
    """
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        
        centroids = new_centroids

    return centroids, labels

def compute_inertia(X, k_range):
    inertias = []
    for k in k_range:
        centroids, labels = kmeans(X, k=k)
        inertia = np.sum((X - centroids[labels])**2)
        inertias.append(inertia)
    return inertias
