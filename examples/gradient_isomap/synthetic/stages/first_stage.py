"""
Stage 1: Manifold Learning for Synthetic Geometries

This script performs manifold learning on synthetic geometries (swiss_roll, s_curve, torus, sphere)
following the same pipeline as MNIST:
1. Generate synthetic data with noise
2. Split into train/val/test
3. Apply Farthest Point Sampling (FPS) to select basis points
4. Train GradientIsomap on basis points
5. Compute projections for all data points
6. Save results for graph regularization training
"""

import os
import sys
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
from utils.cache_utils import load_or_compute_fps, set_global_seed
from data.synthetic_geometries import geometries, noisy_manifold

RANDOM_SEED = 42

def process_geometry(geometry_name, n_samples=1000, n_basis_points=200,
                     noise_percent=0, latent_dim=2, epochs=500,
                     device='cuda', save_checkpoint_history=False):
    """
    Process a single geometry through the manifold learning pipeline.

    Parameters:
    -----------
    geometry_name : str
        Name of the geometry ('swiss_roll', 's_curve', 'torus', 'sphere')
    n_samples : int
        Total number of points to generate
    n_basis_points : int
        Number of basis points to select via FPS (typically 20% of n_samples)
    noise_percent : float
        Noise level as percentage (0.075 = 7.5%)
    latent_dim : int
        Intrinsic dimension of the manifold (2 for most synthetic geometries)
    epochs : int
        Number of epochs for GradientIsomap training
    device : str
        Device to use ('cuda' or 'cpu')
    """

    print(f"\n{'='*80}")
    print(f"PROCESSING GEOMETRY: {geometry_name.upper()}")
    print(f"{'='*80}\n")

    set_global_seed(RANDOM_SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(script_dir, '..'))
    project_root = os.path.abspath(os.path.join(script_dir, '../../../..'))

    outputs_dir = os.path.join(experiment_dir, 'outputs')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_folder = os.path.join(outputs_dir, f'{geometry_name}_run_{timestamp}_n{n_samples}')
    os.makedirs(working_folder, exist_ok=True)


    print(f"Working folder: {working_folder}")

    print(f"Generating {n_samples} points for {geometry_name} with {noise_percent*100}% noise...")
    base_func = geometries[geometry_name][0]
    X, colors = noisy_manifold(base_func, noise_percent=noise_percent, n_samples=n_samples)
    y = colors

    print(f"  Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    print("\nSplitting data into train/val/test (70%/15%/15%)...")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42
    )

    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\n=== FPS SAMPLING ===")
    fps_indices = load_or_compute_fps(
        output_dir=working_folder,
        train_features=X_train,
        num_basis=n_basis_points,
        fps_function=lambda x, n: memory_efficient_fps(x if isinstance(x, np.ndarray) else x.numpy(), n, batch_size=500)
    )

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
        n_neighbors=25,
        checkpoint_each=100,
        save_checkpoint_history=save_checkpoint_history,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=epochs,
        stop_criteria_value=0.001
    )

    isomap.train()
    isomap.visualize_trained()

    best_distances_matrix = isomap.best_distances_matrix
    proj_features = isomap.best_isomap_model()
    base_projections = proj_features.detach().cpu().numpy()

    np.save(os.path.join(working_folder, 'base_projections.npy'), base_projections)
    np.save(os.path.join(working_folder, 'best_distance_matrix.npy'), best_distances_matrix)
    print(f"Saved distance matrix and base projections")
    print(f"Latent_dim: {latent_dim} (saved in metadata)")
    print("\n=== COMPUTING PROJECTIONS ===")
    train_proj_path = os.path.join(working_folder, 'train_projections.npy')


    X_basis = X_train[fps_indices]
    Y_basis = base_projections


    print("Computing projections for training data...")
    projector = Projector(
        source_data=X_train,
        basis_indices=fps_indices,
        upper_triangular_distances=best_distances_matrix,
        n_neighbors=25,
        method='random_forest',
        batch_size=1024,
        precomputed_base_projections=base_projections,
        verbose=True
    )

    projector.compute_projection()
    train_projections = projector.projection
    np.save(train_proj_path, train_projections)
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

    return {'working_folder': working_folder}


def synthetic_manifold_learning_pipeline():
    """
    Main pipeline function for use by run_pipeline.py
    """
    geometries_to_process = [
        'torus',
        'sphere',
        'swiss_roll',
        'swiss_hole',
        's_curve',
        'pseudosphere',
        'hyperboloid',
        'helicoid',
        'multi_scale_torus',
        'nonuniform_sphere',
        'cone_surface',
        'genus_2_surface',
        'connected_multiscale_manifold',
    ]
    n_samples = 10000
    n_basis_points = 2000
    noise_percent = 0.05
    latent_dim = 2
    epochs = 20000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("SYNTHETIC GEOMETRY MANIFOLD LEARNING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Geometries: {geometries_to_process}")
    print(f"  Total samples per geometry: {n_samples}")
    print(f"  Basis points (FPS): {n_basis_points}")
    print(f"  Noise level: {noise_percent * 100}%")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  GradientIsomap epochs: {epochs}")
    print(f"  Device: {device}")

    geometry_folders = {}
    for geom in geometries_to_process:
        try:
            result = process_geometry(
                geometry_name=geom,
                n_samples=n_samples,
                n_basis_points=n_basis_points,
                noise_percent=noise_percent,
                latent_dim=latent_dim,
                epochs=epochs,
                device=device,
                save_checkpoint_history=False
            )
            geometry_folders[geom] = result['working_folder']
        except Exception as e:
            print(f"\nError processing {geom}: {e}")
            import traceback
            traceback.print_exc()

    return {'geometry_folders': geometry_folders}


if __name__ == "__main__":
    synthetic_manifold_learning_pipeline()
