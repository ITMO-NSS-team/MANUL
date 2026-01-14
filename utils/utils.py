import os
import numpy as np
import torch
import random
import json
from sklearn.model_selection import train_test_split
from torchvision import datasets


def split_data(X, y, proportions=(0.7, 0.15, 0.15), random_state=42):
    """
    Split data into train, validation, and test sets.

    Parameters:
    X, y: Features and labels
    proportions: tuple of (train_ratio, val_ratio, test_ratio)
    random_state: Random seed for reproducibility

    Returns:
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_ratio, val_ratio, test_ratio = proportions

    # First split: separate test set
    test_size = test_ratio
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from train
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative_ratio,
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def set_global_seed(seed=42):
    """
    Set fixed seed for all random number generators.
    Ensures reproducibility for numpy, torch, and random.

    Args:
        seed: Integer to initialize generators
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to {seed}")


def restore_mnist_splits(data_dir, test_size_outer=0.2, test_size_inner=0.2, random_state=42):
    """
    Restore train/val/test splits for MNIST dataset.

    Args:
        data_dir: Path to MNIST data directory
        test_size_outer: Test set size (default 0.2)
        test_size_inner: Validation set size from trainval (default 0.2)
        random_state: Seed for reproducibility

    Returns:
        Tuple of X_train, X_val, X_test, y_train, y_val, y_test
    """
    set_global_seed(random_state)

    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
    y = mnist_dataset.targets.numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size_outer, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=test_size_inner,
        random_state=random_state, stratify=y_trainval
    )

    print(f"Restored MNIST splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def restore_data_from_metadata(experiment_folder, project_root=None):
    """
    Universal function for restoring data from metadata.

    Reads experiment_metadata.json and regenerates train/val/test splits
    using the same random seed and parameters as first_stage.

    Args:
        experiment_folder: Path to experiment folder
        project_root: Root directory of project (for MNIST data path).
                      If None, automatically computed as experiment_folder + '../../../..'

    Returns:
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """

    metadata_path = os.path.join(experiment_folder, 'experiment_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    dataset_type = metadata.get('dataset_type', 'mnist')
    random_seed = metadata.get('random_seed', 42)
    split_params = metadata.get('split_params', {})

    if dataset_type == 'mnist':
        if project_root is None:
            project_root = os.path.abspath(os.path.join(experiment_folder, '../../../..'))
        data_dir = os.path.join(project_root, 'data')

        X_train, X_val, X_test, y_train, y_val, y_test = restore_mnist_splits(
            data_dir=data_dir,
            test_size_outer=split_params.get('test_size_outer', 0.2),
            test_size_inner=split_params.get('test_size_inner', 0.2),
            random_state=random_seed
        )

    elif dataset_type == 'synthetic':
        X_train, X_val, X_test, y_train, y_val, y_test = restore_synthetic_splits(
            geometry_name=metadata['geometry_name'],
            n_samples=metadata['n_samples'],
            noise_percent=metadata['noise_percent'],
            random_state=random_seed
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }


def load_data_from_folder(folder_path: str) -> dict:
    """
    Load all necessary data from experiment folder created by first_stage.

    Loads both preprocessed splits (X_train, y_train, etc.) and manifold learning artifacts
    (distance matrix, projections, FPS indices) by restoring from metadata.

    Args:
        folder_path: Path to the folder with saved experiment data

    Returns:
        Dictionary containing:
            - X_train, X_val, X_test: feature arrays
            - y_train, y_val, y_test: target arrays
            - best_distances_matrix: upper triangular distance matrix from GradientIsomap
            - fps_indices: indices of FPS-selected basis points
            - latent_dim: intrinsic dimensionality
            - base_projections: projections for basis points
            - train_projections: projections for all training data
            - folder_path: original folder path
    """
    print(f"\nLoading data from {folder_path}...")

    # Restore train/val/test splits from metadata
    restored_data = restore_data_from_metadata(folder_path)
    X_train = restored_data['X_train']
    X_val = restored_data['X_val']
    X_test = restored_data['X_test']
    y_train = restored_data['y_train']
    y_val = restored_data['y_val']
    y_test = restored_data['y_test']

    # Load manifold learning artifacts
    best_distances_matrix = np.load(os.path.join(folder_path, 'best_distance_matrix.npy'))
    fps_indices = np.load(os.path.join(folder_path, 'fps_indices.npy'))
    base_projections = np.load(os.path.join(folder_path, 'base_projections.npy'))
    train_projections = np.load(os.path.join(folder_path, 'train_projections.npy'))

    # Load latent_dim from metadata
    metadata_path = os.path.join(folder_path, 'experiment_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    latent_dim = int(metadata['latent_dim'])

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  FPS indices: {len(fps_indices)} basis points")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Base projections: {base_projections.shape}")
    print(f"  Train projections: {train_projections.shape}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'best_distances_matrix': best_distances_matrix,
        'fps_indices': fps_indices,
        'latent_dim': latent_dim,
        'base_projections': base_projections,
        'train_projections': train_projections,
        'folder_path': folder_path
    }
