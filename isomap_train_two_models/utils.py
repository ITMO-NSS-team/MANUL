import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA


def farthest_point_sampling(dist_matrix, retain_points):
    """
    Farthest-Point Sampling to select retain_points points that cover the distance matrix.

    Args:
        dist_matrix (torch.Tensor): Full distance matrix (n x n).
        k (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the selected points (k,).
    """
    n = dist_matrix.shape[0]
    
    # Initialize storage for selected indices
    selected_indices = torch.zeros(retain_points, dtype=torch.long)
    
    # Start with a random point
    selected_indices[0] = torch.randint(0, n, (1,))
    
    # Track minimum distances to the current selected points
    min_distances = dist_matrix[selected_indices[0], :]
    
    for i in range(1, retain_points):
        # Select the farthest point from the current selected points
        farthest_point = torch.argmax(min_distances)
        selected_indices[i] = farthest_point
        
        # Update minimum distances to the selected points
        min_distances = torch.minimum(min_distances, dist_matrix[farthest_point, :])
    
    return selected_indices


def reduce_dist_fps(dist_matrix, retain_points):
    pts = farthest_point_sampling(dist_matrix, retain_points)
    reduced_dist = dist_matrix[pts][:, pts]
    return pts, reduced_dist


def plot_points_with_PCA(points, labels):
    labels = np.argmax(labels, axis=1)
    points_2d = PCA(n_components=2).fit_transform(points)
    plt.scatter(points_2d[:, 1], points_2d[:, 0], c=labels)
    plt.show()
