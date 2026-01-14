import os
import time
from datetime import datetime

import numpy as np
import torch
from torchvision import datasets

from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps
from utils.utils import split_data


def mnist_manifold_learning(mnist_folder):
    n_samples = 60000  # Number of images to use from dataset
    n_base_points = 1000
    epochs = 20000  # Number of total epochs for GradientIsomap training (early stopping exists)
    proj_method = 'random_forest'
    device = 'cuda'
    detail_analyse_dimensionality = False

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
    print('Detailed analysis of  intrinsic dimensionality')
    if detail_analyse_dimensionality:
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









