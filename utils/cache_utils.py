import os
import numpy as np
import torch
from typing import Optional, Tuple, Callable, Dict

def load_or_compute_fps(output_dir: str,
                        train_features: np.ndarray,
                        num_basis: int,
                        fps_function: Callable) -> np.ndarray:
    """Loads FPS indices from cache or computes them.

    Args:
        output_dir: Directory where FPS indices are/will be saved
        train_features: Training features for FPS sampling
        num_basis: Number of basis points to select
        fps_function: Function to compute FPS (e.g., fps_torch)

    Returns:
        FPS indices array
    """
    fps_path = os.path.join(output_dir, 'fps_indices.npy')

    if os.path.exists(fps_path):
        print(f"Found cached FPS indices at {fps_path}")
        print(f"Loading {num_basis} basis points")
        return np.load(fps_path)
    else:
        print(f"Computing FPS sampling for {num_basis} basis points...")
        fps_indices = fps_function(torch.tensor(train_features), num_basis)
        np.save(fps_path, fps_indices)
        print(f"Saved FPS indices to {fps_path}")
        return fps_indices


def load_or_train_isomap(output_dir: str,
                         basis_features: np.ndarray,
                         basis_labels: np.ndarray,
                         latent_dim: int,
                         epochs: int,
                         isomap_class) -> Tuple:
    """Loads distance matrix from cache or trains Isomap.

    Args:
        output_dir: Directory where distance matrix is/will be saved
        basis_features: Basis point features
        basis_labels: Basis point labels
        latent_dim: Latent dimension of manifold
        epochs: Number of training epochs
        isomap_class: GradientIsomap class

    Returns:
        Tuple of (weights_matrix, distance_matrix)
    """
    distance_matrix_path = os.path.join(output_dir, 'best_distance_matrix.npy')

    if os.path.exists(distance_matrix_path):
        print(f"Found cached distance matrix at {distance_matrix_path}")
        print(f"Skipping Isomap training ({epochs} epochs saved)")
        distance_matrix = np.load(distance_matrix_path)
        weights_matrix = distance_matrix_to_weights(distance_matrix)
        return weights_matrix, distance_matrix
    else:
        print(f"Starting Isomap training for {epochs} epochs...")
        isomap = isomap_class(
            train_feature=basis_features,
            train_target=basis_labels,
            latent_len=latent_dim,
            epochs=epochs
        )
        isomap.train()
        weights_matrix, distance_matrix = isomap.get_weights_matrix()
        np.save(distance_matrix_path, distance_matrix)
        print(f"Saved distance matrix to {distance_matrix_path}")
        return weights_matrix, distance_matrix


def load_or_compute_projections(output_dir: str,
                                data_dict: Dict[str, np.ndarray],
                                weights_matrix,
                                fps_indices: np.ndarray,
                                projector_class,
                                method: str = 'ensemble_knn') -> Dict[str, np.ndarray]:
    """Loads projections from cache or computes them for train/val/test splits.

    Args:
        output_dir: Directory where projections are/will be saved
        data_dict: Dictionary with split names as keys and data as values
        weights_matrix: Distance matrix weights
        fps_indices: FPS basis indices
        projector_class: Projector class
        method: Projection method to use

    Returns:
        Dictionary of projections for each split
    """
    projections = {}

    for split_name, split_data in data_dict.items():
        proj_path = os.path.join(output_dir, f'{split_name}_projections.npy')

        if os.path.exists(proj_path):
            print(f"Found cached {split_name} projections")
            projections[split_name] = np.load(proj_path)
        else:
            print(f"Computing projections for {split_name} data...")
            projector = projector_class(
                source_data=split_data,
                weights_matrix=weights_matrix,
                basis_indices=fps_indices,
                method=method
            )
            proj = projector.compute_all_projections()
            np.save(proj_path, proj)
            print(f"Saved {split_name} projections")
            projections[split_name] = proj

    return projections


def distance_matrix_to_weights(distance_matrix: np.ndarray) -> list:
    """Reconstructs weights_matrix from distance_matrix.

    Converts symmetric distance matrix to lower triangular format
    used by the framework.

    Args:
        distance_matrix: Square symmetric distance matrix [N, N]

    Returns:
        weights_matrix: List of arrays (lower triangular format)
    """
    weights_matrix = []
    for i in range(1, distance_matrix.shape[0]):
        weights_matrix.append(distance_matrix[i, :i])
    return weights_matrix


def check_required_files(input_dir: str, required_files: list) -> bool:
    """Checks if all required files exist in directory.

    Args:
        input_dir: Directory to check
        required_files: List of required filenames

    Returns:
        True if all files exist, False otherwise
    """
    missing = [f for f in required_files if not os.path.exists(os.path.join(input_dir, f))]

    if missing:
        print(f"Error: Missing required files from Stage 1:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print(f"All required files found in {input_dir}")
        return True
