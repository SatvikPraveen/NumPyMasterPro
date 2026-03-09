"""
Unit tests for kmeans_utils module
"""
import pytest
import numpy as np
from scripts.kmeans_utils import (
    generate_data,
    initialize_centroids,
    assign_clusters,
    update_centroids,
    kmeans,
    compute_inertia,
    compute_cluster_inertia,
)


class TestGenerateData:
    """Tests for generate_data function"""
    
    def test_data_shape(self):
        data = generate_data()
        assert data.shape == (150, 2)
    
    def test_data_type(self):
        data = generate_data()
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float64


class TestInitializeCentroids:
    """Tests for initialize_centroids function"""
    
    def test_centroid_count(self):
        data = np.random.rand(100, 2)
        centroids = initialize_centroids(data, k=3)
        assert centroids.shape == (3, 2)
    
    def test_centroids_from_data(self):
        data = np.random.rand(100, 2)
        centroids = initialize_centroids(data, k=5)
        
        # Centroids should be within the range of data
        assert centroids.min() >= data.min()
        assert centroids.max() <= data.max()
    
    def test_invalid_k(self):
        data = np.random.rand(10, 2)
        with pytest.raises(ValueError):
            initialize_centroids(data, k=15)  # k > n_samples


class TestAssignClusters:
    """Tests for assign_clusters function"""
    
    def test_cluster_assignment_shape(self):
        data = np.random.rand(100, 2)
        centroids = np.random.rand(3, 2)
        labels = assign_clusters(data, centroids)
        
        assert labels.shape == (100,)
    
    def test_cluster_labels_range(self):
        data = np.random.rand(100, 2)
        centroids = np.random.rand(3, 2)
        labels = assign_clusters(data, centroids)
        
        # Labels should be in range [0, k-1]
        assert labels.min() >= 0
        assert labels.max() <= 2
    
    def test_simple_case(self):
        # Create simple data with clear clusters
        data = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        labels = assign_clusters(data, centroids)
        
        # First two points should go to cluster 0, last two to cluster 1
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


class TestUpdateCentroids:
    """Tests for update_centroids function"""
    
    def test_centroid_update_shape(self):
        data = np.random.rand(100, 2)
        labels = np.random.randint(0, 3, 100)
        centroids = update_centroids(data, labels, k=3)
        
        assert centroids.shape == (3, 2)
    
    def test_centroid_calculation(self):
        # Simple test case
        data = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
        labels = np.array([0, 0, 1, 1])
        centroids = update_centroids(data, labels, k=2)
        
        # Centroid 0 should be mean of [0,0] and [2,2] = [1,1]
        # Centroid 1 should be mean of [10,10] and [12,12] = [11,11]
        assert np.allclose(centroids[0], [1, 1])
        assert np.allclose(centroids[1], [11, 11])


class TestKmeans:
    """Tests for kmeans function"""
    
    def test_kmeans_basic(self):
        data = generate_data()
        centroids, labels = kmeans(data, k=3, max_iters=100, tol=1e-4)
        
        assert centroids.shape == (3, 2)
        assert labels.shape == (150,)
        assert labels.min() >= 0
        assert labels.max() <= 2
    
    def test_kmeans_convergence(self):
        # Simple data that should converge quickly
        data = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        centroids, labels = kmeans(data, k=2, max_iters=10)
        
        # Should create two clusters
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 2
    
    def test_kmeans_with_random_seed(self):
        data = generate_data()
        np.random.seed(42)
        centroids1, labels1 = kmeans(data, k=3)
        
        np.random.seed(42)
        centroids2, labels2 = kmeans(data, k=3)
        
        # Results should be identical with same seed
        assert np.allclose(centroids1, centroids2)


class TestComputeInertia:
    """Tests for compute_cluster_inertia function"""
    
    def test_inertia_positive(self):
        data = np.random.rand(100, 2)
        centroids = np.random.rand(3, 2)
        labels = np.random.randint(0, 3, 100)
        
        inertia = compute_cluster_inertia(data, centroids, labels)
        assert inertia >= 0
    
    def test_inertia_zero_for_perfect_clusters(self):
        # Data points exactly at centroids
        centroids = np.array([[0, 0], [10, 10]])
        data = centroids.copy()
        labels = np.array([0, 1])
        
        inertia = compute_cluster_inertia(data, centroids, labels)
        assert np.isclose(inertia, 0.0)
    
    def test_inertia_calculation(self):
        # Manual calculation test
        data = np.array([[0, 0], [2, 0], [10, 0], [12, 0]])
        centroids = np.array([[1, 0], [11, 0]])
        labels = np.array([0, 0, 1, 1])
        
        # Inertia = (0-1)^2 + (2-1)^2 + (10-11)^2 + (12-11)^2 = 1 + 1 + 1 + 1 = 4
        inertia = compute_cluster_inertia(data, centroids, labels)
        assert np.isclose(inertia, 4.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
