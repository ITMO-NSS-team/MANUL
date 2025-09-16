import numpy as np
import torch


def farthest_point_sampling(dist_matrix, retain_points):
    """
    Farthest-Point Sampling to select retain_points points that cover the distance matrix.

    Args:
        dist_matrix (torch.Tensor): Full distance matrix (n x n).
        retain_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the selected points (retain_points,).
    """
    n = dist_matrix.shape[0]
    selected_indices = torch.zeros(retain_points, dtype=torch.long)
    selected_indices[0] = torch.randint(0, n, (1,))
    min_distances = dist_matrix[selected_indices[0], :]

    for i in range(1, retain_points):
        farthest_point = torch.argmax(min_distances)
        selected_indices[i] = farthest_point
        min_distances = torch.minimum(min_distances, dist_matrix[farthest_point, :])

    return selected_indices


def reduce_dist_fps(dist_matrix, retain_points):
    """
    Function for sparce distance matrix with FPS
    """
    pts = farthest_point_sampling(dist_matrix, retain_points)
    reduced_dist = dist_matrix[pts][:, pts]
    return pts, reduced_dist


def memory_efficient_fps(features, n_samples, batch_size=1000):
    """
    Memory-efficient Farthest Point Sampling using batch processing
    """
    n_points = features.shape[0]
    selected_indices = []
    current_point = np.random.randint(0, n_points)
    selected_indices.append(current_point)
    min_distances = np.linalg.norm(features - features[current_point], axis=1)

    while len(selected_indices) < n_samples:
        print(f"Selected {len(selected_indices)}/{n_samples} points", end='\r')
        farthest_point = np.argmax(min_distances)
        selected_indices.append(farthest_point)

        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_distances = np.linalg.norm(features[i:end_idx] - features[farthest_point], axis=1)
            min_distances[i:end_idx] = np.minimum(min_distances[i:end_idx], batch_distances)

    return selected_indices
