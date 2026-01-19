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
from data.synthetic_geometries import geometries, noisy_manifold


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
    n_base_points = 1000  # Number of base points to select via FPS (typically 20% of n_samples)
    noise_percent = 0.01  # Noise level as percentage (0.075 = 7.5%)
    latent_dim = 2  # Intrinsic dimension of the manifold (2 for most synthetic geometries)
    epochs = 5000  # Number of total epochs for GradientIsomap training (early stopping exists)
    proj_method = 'random_forest'
    device = 'cuda'

    print("=" * 80)
    print("SYNTHETIC GEOMETRY MANIFOLD LEARNING PIPELINE")
    print("=" * 80)

    print(f"Working folder: {working_folder}")
    print(f"Generating {n_samples} points for {geometry_name} with {noise_percent * 100}% noise...")
    geometry_function = geometries[geometry_name][0]
    X, y = noisy_manifold(geometry_function, noise_percent=noise_percent, n_samples=n_samples)
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
        plot_convergence=False,  # Show convergence plot for each intrinsic approximation NN during Isomap optimization
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
                      'Noise level', 'Latent dimension', 'Device', 'FPS time', 'Isomap train time',
                      'Projection method', 'Projection time'],
        'Value': [geometry_name, n_samples, n_base_points,
                  noise_percent, latent_dim, device, fps_extract_time, isomap_train_time,
                  proj_method, projection_time]
    })
    metadata.to_csv(f'{working_folder}/metadata.csv', index=False)
    print("\nConfiguration:")
    for _, row in metadata.iterrows():
        print(f"{row['Parameter']} - {row['Value']}")
    print(f"Geometry {geometry_name} processing complete!")

    return working_folder


if __name__ == "__main__":
    outputs_dir = 'outputs/'
    geometry_name = 'torus'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_folder = os.path.join(outputs_dir, f'{geometry_name}_run_{timestamp}')
    os.makedirs(working_folder, exist_ok=True)

    output_folder = synthetic_manifold_learning_pipeline(geometry_name, working_folder)
