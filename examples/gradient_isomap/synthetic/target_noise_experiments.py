import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import time

from Adam.GradientIsomap import GradientIsomap
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.utils import split_data
from data.synthetic_geometries import geometries


def target_noise_wrapper(base_func, noise_percent=0.0, n_samples=1000):
    """
    Add Gaussian noise to target values.

    Args:
        base_func: Geometry function from synthetic_geometries
        noise_percent: Noise level (0.01 = 1%)
        n_samples: Number of samples

    Returns:
        points: Clean coordinates
        colors: Noisy targets
    """
    points, colors = base_func(n_samples=n_samples)

    if noise_percent > 0:
        noise = np.random.normal(0, np.max(colors) * noise_percent, colors.shape)
        colors = colors + noise

    return points, colors


def synthetic_target_noise_pipeline(geometry_name, working_folder, noise_level=0.01):
    """
    Manifold learning pipeline with target noise.
    Parameters:
    -----------
    geometry_name : str
        Name of geometry ('swiss_roll', 's_curve', 'torus', 'sphere', etc.)
    working_folder : str
        Path to outputs directory
    noise_level : float
        Noise percentage for target (0.01 = 1%)
    """

    n_samples = 5000
    n_base_points = 1000
    latent_dim = 2
    epochs = 5000
    proj_method = 'random_forest'
    device = 'cuda'

    print("=" * 80)
    print("TARGET NOISE MANIFOLD LEARNING PIPELINE")
    print("=" * 80)

    print(f"Working folder: {working_folder}")
    print(f"Generating {n_samples} points for {geometry_name}")
    print(f"Target noise: {noise_level * 100}% (coordinates are clean)")

    geometry_function = geometries[geometry_name][0]
    X, y = target_noise_wrapper(geometry_function, noise_percent=noise_level, n_samples=n_samples)

    np.save(f'{working_folder}/all_features.npy', X)
    np.save(f'{working_folder}/all_targets.npy', y)

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
        fps_extract_time = 0
        print(f'FPS indices loaded from {working_folder}/fps_indices.npy')
    else:
        start_time = time.time()
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(f'{working_folder}/fps_indices.npy', fps_indices)
        fps_extract_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'FPS indices saved to {working_folder}/fps_indices.npy')

    print("\n=== MANIFOLD LEARNING ===")
    train_features = torch.tensor(X_train[fps_indices], dtype=torch.float32).to(device)
    train_target = torch.tensor(y_train[fps_indices], dtype=torch.float32).to(device)

    print(f"Training GradientIsomap (latent_dim={latent_dim}, epochs={epochs})...")
    start_time = time.time()
    isomap = GradientIsomap(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_dim,
        checkpoint_each=100,
        save_checkpoint_matrix=False,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=epochs,
        stop_criteria_value=0.001
    )
    isomap.train()
    isomap_train_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    isomap.visualize_trained()

    base_projection = isomap.best_isomap_model().detach().cpu().numpy()
    np.save(f'{working_folder}/base_projection.npy', base_projection)
    print(f"Saved base projections {working_folder}/base_projection.npy")

    print("\n=== COMPUTING PROJECTIONS ===")
    start_time = time.time()
    projector = Projector(
        source_data=X_train,
        base_indices=fps_indices,
        batch_size=1024,
        base_projection=base_projection,
    )
    train_projections = projector.compute_projection(method=proj_method)
    projection_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    np.save(os.path.join(working_folder, 'train_projections.npy'), train_projections)
    print(f"Saved train projections")

    metadata = pd.DataFrame({
        'Parameter': ['Geometry name', 'Total samples', 'Base points (FPS)',
                      'Noise type', 'Noise level', 'Latent dimension', 'Device',
                      'FPS time', 'Isomap train time', 'Projection method', 'Projection time'],
        'Value': [geometry_name, n_samples, n_base_points,
                  'target', noise_level, latent_dim, device,
                  fps_extract_time, isomap_train_time, proj_method, projection_time]
    })
    metadata.to_csv(f'{working_folder}/metadata.csv', index=False)

    print("\nConfiguration:")
    for _, row in metadata.iterrows():
        print(f"{row['Parameter']} - {row['Value']}")
    print(f"Geometry {geometry_name} processing complete!")

    return working_folder


if __name__ == "__main__":
    outputs_dir = 'outputs_target_noise/'

    geometries_to_test = [
        'sphere', 'torus', 'swiss_roll', 'swiss_hole', 's_curve',
        'pseudosphere', 'hyperboloid', 'helicoid', 'multi_scale_torus',
        'nonuniform_sphere', 'cone_surface', 'genus_2_surface',
        'connected_multiscale_manifold'
    ]

    noise_levels = [0.01, 0.03, 0.05, 0.1]
    n_runs = 5

    print(f"\n{'=' * 80}")
    print("TARGET NOISE EXPERIMENT")
    print(f"{'=' * 80}")
    print(f"Geometries: {len(geometries_to_test)}")
    print(f"Noise levels: {noise_levels}")
    print(f"Runs per combination: {n_runs}")
    print(f"Total experiments: {len(geometries_to_test) * len(noise_levels) * n_runs}")
    print(f"{'=' * 80}\n")

    for geom in geometries_to_test:
        for noise in noise_levels:
            for run in range(n_runs):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                working_folder = f'{outputs_dir}/{geom}/noise_{noise}/{geom}_run{run}_{timestamp}'
                os.makedirs(working_folder, exist_ok=True)

                print(f"\n>>> Geometry: {geom}, Noise: {noise * 100}%, Run: {run + 1}/{n_runs}")

                try:
                    synthetic_target_noise_pipeline(geom, working_folder, noise_level=noise)
                    print(f"✓ Completed")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

    print(f"\n{'=' * 80}")
    print("All experiments saved to outputs_target_noise folder")
    print(f"{'=' * 80}")