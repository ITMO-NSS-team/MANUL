"""
Stage 1: Manifold Learning for Synthetic Geometries

This script performs manifold learning on synthetic geometries (swiss_roll, s_curve, torus, sphere)
following the same pipeline as MNIST:
1. Generate synthetic data with noise
2. Split into train/val/test
3. Apply Farthest Point Sampling (FPS) to select base points
4. Train GradientIsomap on base points
5. Compute projections for all data points
6. Save results for graph regularization training
"""

import os
from datetime import datetime
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import json
import matplotlib

matplotlib.use('Agg')

from Adam.GradientIsomap import GradientIsomap
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.utils import set_global_seed, split_data
from data.synthetic_geometries import geometries, noisy_manifold

RANDOM_SEED = 42


def synthetic_manifold_learning_pipeline(geometry_name, working_folder):
    """
        Process a single geometry through the manifold learning pipeline.

        Parameters:
        -----------
        geometry_name : str
            Name of the geometry ('swiss_roll', 's_curve', 'torus', 'sphere')
        working_folder : str
            Path to outputs directory
    """

    n_samples = 5000  # Total number of points to generate
    n_base_points = 500  # Number of base points to select via FPS (typically 20% of n_samples)
    noise_percent = 0.05  # Noise level as percentage (0.075 = 7.5%)
    latent_dim = 2  # Intrinsic dimension of the manifold (2 for most synthetic geometries)
    epochs = 20000  # Number of epochs for GradientIsomap training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("SYNTHETIC GEOMETRY MANIFOLD LEARNING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Geometry name: {geometry_name}")
    print(f"  Total samples per geometry: {n_samples}")
    print(f"  Base points (FPS): {n_base_points}")
    print(f"  Noise level: {noise_percent * 100}%")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  GradientIsomap epochs: {epochs}")
    print(f"  Device: {device}")

    set_global_seed(RANDOM_SEED)

    print(f"Working folder: {working_folder}")
    print(f"Generating {n_samples} points for {geometry_name} with {noise_percent * 100}% noise...")
    base_func = geometries[geometry_name][0]
    X, colors = noisy_manifold(base_func, noise_percent=noise_percent, n_samples=n_samples)
    y = colors

    print(f"  Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    print("\nSplitting data into train/val/test (70%/15%/15%)...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, (0.7, 0.15, 0.15))

    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n=== FPS SAMPLING ===")
    if os.path.exists(f'{working_folder}/fps_indices.npy'):
        fps_indices = np.load(f'{working_folder}/fps_indices.npy')
        print(f'FPS indices loaded from {working_folder}/fps_indices.npy')
    else:
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(f'{working_folder}/fps_indices.npy', fps_indices)
        print(f'FPS indices saved to {working_folder}/fps_indices.npy')

    X_train_sparse = X_train[fps_indices]
    y_train_sparse = y_train[fps_indices]

    print("\n=== MANIFOLD LEARNING ===")
    train_features = torch.tensor(X_train_sparse, dtype=torch.float32).to(device)
    train_target = torch.tensor(y_train_sparse, dtype=torch.float32).to(device)

    print(f"Training GradientIsomap (latent_dim={latent_dim}, epochs={epochs})...")
    isomap = GradientIsomap(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_dim,
        checkpoint_each=100,
        save_checkpoint_matrix=True,
        logs_folder=working_folder,
        plot_convergence=False,  # Show convergence plot for each intrinsic approximation NN during Isomap optimization
        epochs=epochs,
        stop_criteria_value=0.001
    )

    isomap.train()
    isomap.visualize_trained()

    best_distances_matrix = isomap.best_distances_matrix
    np.save(f'{working_folder}/best_distance_matrix.npy', best_distances_matrix)
    print(f"Saved distance matrix {working_folder}/best_distance_matrix.npy")

    proj_features = isomap.best_isomap_model()
    base_projections = proj_features.detach().cpu().numpy()
    np.save(f'{working_folder}/base_projections.npy', base_projections)
    print(f"Saved base projections {working_folder}/base_projections.npy")

    print("\n=== COMPUTING PROJECTIONS ===")
    projector = Projector(
        source_data=X_train,
        base_indices=fps_indices,
        upper_triangular_distances=best_distances_matrix,
        method='random_forest',
        batch_size=1024,
        precomputed_base_projections=base_projections,
        verbose=True
    )

    projector.compute_projection()
    train_projections = projector.projection
    np.save(os.path.join(working_folder, 'train_projections.npy'), train_projections)
    print(f"Saved train projections")

    experiment_metadata = {
        'dataset_type': 'synthetic',
        'random_seed': RANDOM_SEED,
        'geometry_name': geometry_name,
        'n_samples': n_samples,
        'noise_percent': noise_percent,
        'latent_dim': int(latent_dim),
        'split_params': {
            'test_size_outer': 0.15,
            'test_size_inner': 0.176
        }
    }
    metadata_path = os.path.join(working_folder,
                                 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    print(f"Saved experiment metadata to {metadata_path}")
    print(f"Geometry {geometry_name} processing complete!")

    return working_folder


if __name__ == "__main__":
    outputs_dir = ''
    geometry_name = 'torus'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_folder = os.path.join(outputs_dir, f'{geometry_name}_run_{timestamp}')
    os.makedirs(working_folder, exist_ok=True)

    output_folder = synthetic_manifold_learning_pipeline(geometry_name, working_folder)
