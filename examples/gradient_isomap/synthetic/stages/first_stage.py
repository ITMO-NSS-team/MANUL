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

import matplotlib
matplotlib.use('Agg')

from Adam.GradientIsomap import GradientIsomap
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.cache_utils import load_or_compute_fps
from data.synthetic_geometries import geometries, noisy_manifold



def process_geometry(geometry_name, n_samples=1000, n_basis_points=200,
                     noise_percent=0, latent_dim=2, epochs=500,
                     device='cuda'):
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

    distance_matrix_path = os.path.join(working_folder, 'best_distance_matrix.npy')

    if os.path.exists(distance_matrix_path):
        print(f"Found cached distance matrix at {distance_matrix_path}")
        print(f"Skipping Isomap training ({epochs} epochs saved)")
        best_distances_matrix = np.load(distance_matrix_path)

        n_basis = len(fps_indices)
        weights_matrix = np.zeros((n_basis, n_basis))
        idx = 0
        for i in range(n_basis):
            for j in range(i+1, n_basis):
                weights_matrix[i, j] = best_distances_matrix[idx]
                weights_matrix[j, i] = best_distances_matrix[idx]
                idx += 1

        base_proj_path = os.path.join(working_folder, 'base_projections.npy')
        base_projections = np.load(base_proj_path)
    else:
        print(f"Training GradientIsomap (latent_dim={latent_dim}, epochs={epochs})...")
        isomap = GradientIsomap(
            train_feature=train_features,
            train_target=train_target,
            latent_len=latent_dim,
            checkpoint_each=100,
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

        print(f"  Best distance matrix size: {best_distances_matrix.shape}")
        print(f"  Base projections shape: {base_projections.shape}")

        print("\nReconstructing full distance matrix...")
        n_basis = len(fps_indices)
        weights_matrix = np.zeros((n_basis, n_basis))
        idx = 0
        for i in range(n_basis):
            for j in range(i+1, n_basis):
                weights_matrix[i, j] = best_distances_matrix[idx]
                weights_matrix[j, i] = best_distances_matrix[idx]
                idx += 1
        print(f"  Reconstructed distance matrix: {weights_matrix.shape}")

        np.save(os.path.join(working_folder, 'base_projections.npy'), base_projections)
        np.save(os.path.join(working_folder, 'best_distance_matrix.npy'), best_distances_matrix)
        print(f"Saved distance matrix to {distance_matrix_path}")

    print("\n=== COMPUTING PROJECTIONS ===")
    train_proj_path = os.path.join(working_folder, 'train_projections.npy')
    val_proj_path = os.path.join(working_folder, 'val_projections.npy')

    X_basis = X_train[fps_indices]
    Y_basis = base_projections

    if os.path.exists(train_proj_path):
        print(f"Found cached train projections")
    else:
        print("Computing projections for training data...")
        projector = Projector(
            source_data=X_train,
            weights_matrix=weights_matrix,
            basis_indices=fps_indices,
            n_neighbors=10,
            method='ensemble_knn',
            batch_size=256,
            precomputed_base_projections=base_projections,
            verbose=True
        )
        projector.compute_all_projections()
        train_projections = projector.all_projections
        np.save(train_proj_path, train_projections)
        print(f"Saved train projections")

    if os.path.exists(val_proj_path):
        print(f"Found cached val projections")
    else:
        print("Computing projections for validation data...")
        from regularizator.GraphRegTrainer import project_ensemble_knn
        val_projections = project_ensemble_knn(X_basis, Y_basis, X_val)
        np.save(val_proj_path, val_projections)
        print(f"Saved val projections")

    print(f"\nSaving data to {working_folder}/...")
    np.save(os.path.join(working_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(working_folder, 'X_val.npy'), X_val)
    np.save(os.path.join(working_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(working_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(working_folder, 'y_val.npy'), y_val)
    np.save(os.path.join(working_folder, 'y_test.npy'), y_test)
    np.save(os.path.join(working_folder, 'latent_dim.npy'), latent_dim)

    print(f"Geometry {geometry_name} processing complete!")

    return {
        'geometry': geometry_name,
        'working_folder': working_folder,
        'latent_dim': latent_dim,
        'n_basis_points': len(fps_indices),
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0]
    }


if __name__ == "__main__":
    geometries_to_process = ['sphere',]
    n_samples = 5000
    n_basis_points = 1000
    noise_percent = 0.05
    latent_dim = 2
    epochs = 10000
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

    results = []
    for geom in geometries_to_process:
        try:
            result = process_geometry(
                geometry_name=geom,
                n_samples=n_samples,
                n_basis_points=n_basis_points,
                noise_percent=noise_percent,
                latent_dim=latent_dim,
                epochs=epochs,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"\nError processing {geom}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for res in results:
        print(f"\n{res['geometry']}:")
        print(f"  Folder: {res['working_folder']}")
        print(f"  Latent dim: {res['latent_dim']}")
        print(f"  Basis points: {res['n_basis_points']}")
        print(f"  Data splits: {res['train_size']} / {res['val_size']} / {res['test_size']}")

    print("\nAll geometries processed successfully!")
    print("Ready for Stage 2: Graph Regularization Training")


def synthetic_manifold_learning_pipeline():
    """
    Main pipeline function for use by run_pipeline.py
    """
    geometries_to_process = ['torus']
    n_samples = 5000
    n_basis_points = 1000
    noise_percent = 0.05
    latent_dim = 2
    epochs = 10000
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
                device=device
            )
            geometry_folders[geom] = result['working_folder']
        except Exception as e:
            print(f"\nError processing {geom}: {e}")
            import traceback
            traceback.print_exc()

    return {'geometry_folders': geometry_folders}
