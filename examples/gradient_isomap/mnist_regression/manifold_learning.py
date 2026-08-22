import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torchvision import datasets

from Adam.GradientIsomap import GradientIsomap
from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.Projector import Projector
from utils.fps_implementation import memory_efficient_fps
from utils.utils import split_data


def mnist_manifold_learning(mnist_folder, latent_dim_override=None):
    n_samples = 60000  # Number of images to use from dataset
    n_base_points = 2000
    epochs = 20000  # Number of total epochs for GradientIsomap training (early stopping exists)
    proj_method = 'random_forest'
    device = 'cuda'
    detail_analyse_dimensionality = True
    project_entire_dataset = False

    mnist_dataset = datasets.MNIST(root='../data', train=True, download=True)

    X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
    y = mnist_dataset.targets.numpy()

    X = X[:n_samples]
    y = y[:n_samples]

    print(f"  Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    print("\nSplitting data into train/val/test (70%/15%/15%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, (0.7, 0.15, 0.15))
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    print('Calculate intrinsic dimensionality')
    analyser = DimensionalityAnalyser()
    latent_dim = analyser.analyse_dimensions(
        X_train,
        method='eigenvalue',
        n_samples=1000)
    print(f'Latent dimensionality calculated: {latent_dim}')
    if latent_dim_override is not None:
        print(f'Overriding calculated latent dimensionality with explicit latent_dim_override={latent_dim_override}')
        latent_dim = latent_dim_override

    if detail_analyse_dimensionality:
        print('Detailed analysis of  intrinsic dimensionality')
        analyser.plot_dimension_histograms(dataset_name="MNIST",
                                           save_path=f'{mnist_folder}/hist_plot.png')
        _, _ = analyser.plot_variance_threshold_analysis(X_train,
                                                         dataset_name="MNIST",
                                                         n_samples=1000,
                                                         save_path=f'{mnist_folder}/variance_plot.png')

    print("\n=== FPS SAMPLING ===")
    if os.path.exists(f'{mnist_folder}/fps_indices.npy'):
        fps_indices = np.load(f'{mnist_folder}/fps_indices.npy')
        fps_extract_time = 0
        print(f'FPS indices loaded from {mnist_folder}/fps_indices.npy')
    else:
        start_time = time.time()
        fps_indices = memory_efficient_fps(features=X_train, n_samples=n_base_points, batch_size=500)
        np.save(f'{mnist_folder}/fps_indices.npy', fps_indices)
        fps_extract_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'FPS indices saved to {mnist_folder}/fps_indices.npy')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_folder = f'{mnist_folder}/mnist_run_{timestamp}'
    os.makedirs(working_folder, exist_ok=True)

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

    projection_time = None
    if project_entire_dataset:
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
        'Parameter': ['Total samples', 'Base points (FPS)',
                      'Latent dimension', 'Device', 'FPS time', 'Isomap train time',
                      'Projection method', 'Projection time'],
        'Value': [n_samples, n_base_points,
                  latent_dim, device, fps_extract_time, isomap_train_time,
                  proj_method, projection_time]
    })
    metadata.to_csv(f'{working_folder}/metadata.csv', index=False)
    print("\nConfiguration:")
    for _, row in metadata.iterrows():
        print(f"{row['Parameter']} - {row['Value']}")
    print(f"MNIST processing complete!")

    return working_folder


if __name__ == "__main__":
    outputs_dir = 'outputs_60000/'
    os.makedirs(outputs_dir, exist_ok=True)
    output_folder = mnist_manifold_learning(outputs_dir)








