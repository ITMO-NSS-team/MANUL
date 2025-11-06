import os
import sys
sys.path.append('../../..')

import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from Adam.GradientIsomap import GradientIsomap
from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.cache_utils import load_or_compute_fps, load_or_train_isomap, load_or_compute_projections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../..'))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def mnist_manifold_learning_example():
    """
    MNIST manifold learning example with FPS sampling and local PCA dimension estimation.
    """
    n_samples = 2000
    latent_len = 300  # precomp latent dimension for Isomap

    working_folder = os.path.join(OUTPUTS_DIR, f'mnist_{n_samples}')
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    print(f"Working folder: {working_folder}")
    print("Loading MNIST data...")

    mnist_raw_path = os.path.join(DATA_DIR, 'MNIST', 'raw')
    mnist_exists = os.path.exists(mnist_raw_path) and len(os.listdir(mnist_raw_path)) > 0

    if mnist_exists:
        print(f"  Found existing MNIST data in {mnist_raw_path}")
    else:
        print(f"  MNIST data not found, will download to {DATA_DIR}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=not mnist_exists, transform=transform)

    X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1)
    y = mnist_dataset.targets.numpy()

    print("Splitting data into train/val/test...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    print("\n=== DIMENSIONALITY ANALYSIS ===")
    analyser = DimensionalityAnalyser()

    analyser.analyse_dimensions(
        X_train,
        method='both',
        n_samples=1000
    )
    analyser.plot_dimension_histograms(dataset_name="MNIST",
                                       save_path=f'{working_folder}/hist_plot.png')
    _, _ = analyser.plot_variance_threshold_analysis(X_train,
                                                      dataset_name="MNIST",
                                                      n_samples=1000,
                                                      save_path=f'{working_folder}/variance_plot.png')
    latent_len = analyser.get_latent_dim(method='eigenvalue')

    print("\n=== FPS SAMPLING ===")
    fps_indices = load_or_compute_fps(
        output_dir=working_folder,
        train_features=X_train,
        num_basis=n_samples,
        fps_function=memory_efficient_fps
    )

    X_train_sparse = X_train[fps_indices]
    y_train_sparse = y_train[fps_indices]

    print("\n=== MANIFOLD LEARNING ===")
    train_features = torch.tensor(X_train_sparse, dtype=torch.float32).to('cuda')
    train_target = torch.tensor(y_train_sparse, dtype=torch.float32).to('cuda')

    distance_matrix_path = os.path.join(working_folder, 'best_distance_matrix.npy')

    if os.path.exists(distance_matrix_path):
        print(f"Found cached distance matrix at {distance_matrix_path}")
        print(f"Skipping Isomap training (15000 epochs saved)")
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
        print(f"Starting Isomap training for 15000 epochs...")
        isomap = GradientIsomap(
            train_feature=train_features,
            train_target=train_target,
            latent_len=latent_len,
            checkpoint_each=100,
            logs_folder=working_folder,
            plot_convergence=False,
            epochs=15000
        )

        isomap.train()
        isomap.visualize_trained()

        best_distances_matrix = isomap.best_distances_matrix
        proj_features = isomap.best_isomap_model()
        base_projections = proj_features.detach().cpu().numpy()

        print("\n=== COMPUTING ALL PROJECTIONS ===")
        n_basis = len(fps_indices)
        weights_matrix = np.zeros((n_basis, n_basis))
        idx = 0
        for i in range(n_basis):
            for j in range(i+1, n_basis):
                weights_matrix[i, j] = best_distances_matrix[idx]
                weights_matrix[j, i] = best_distances_matrix[idx]
                idx += 1
        print(f"Reconstructed distance matrix: {weights_matrix.shape}")

        np.save(f'{working_folder}/base_projections.npy', base_projections)
        np.save(f'{working_folder}/best_distance_matrix.npy', best_distances_matrix)
        print(f"Saved distance matrix to {distance_matrix_path}")

    print("\n=== COMPUTING PROJECTIONS ===")
    train_proj_path = os.path.join(working_folder, 'train_projections.npy')
    val_proj_path = os.path.join(working_folder, 'val_projections.npy')

    X_basis = X_train[fps_indices]
    Y_basis = base_projections

    if os.path.exists(train_proj_path):
        print(f"Found cached train projections")
        train_projections = np.load(train_proj_path)
    else:
        print("Computing projections for training data...")
        projector = Projector(
            source_data=X_train,
            weights_matrix=weights_matrix,
            basis_indices=fps_indices,
            n_neighbors=10,
            method='ensemble_knn',
            batch_size=128,
            precomputed_base_projections=base_projections,
            verbose=True
        )
        projector.compute_all_projections()
        train_projections = projector.all_projections
        np.save(train_proj_path, train_projections)
        print(f"Saved train projections")

    if os.path.exists(val_proj_path):
        print(f"Found cached val projections")
        val_projections = np.load(val_proj_path)
    else:
        print("Computing projections for validation data...")
        from regularizator.GraphRegTrainer import project_ensemble_knn
        val_projections = project_ensemble_knn(X_basis, Y_basis, X_val)
        np.save(val_proj_path, val_projections)
        print(f"Saved val projections")

    print("\n=== SAVING DATA FOR GRAPH REGULARIZATION ===")
    np.save(f'{working_folder}/X_train.npy', X_train)
    np.save(f'{working_folder}/X_val.npy', X_val)
    np.save(f'{working_folder}/X_test.npy', X_test)
    np.save(f'{working_folder}/y_train.npy', y_train)
    np.save(f'{working_folder}/y_val.npy', y_val)
    np.save(f'{working_folder}/y_test.npy', y_test)
    np.save(f'{working_folder}/latent_dim.npy', latent_len)

    print(f"\nAll data saved to {working_folder}/")

    return {
        'working_folder': working_folder,
        'latent_dim': latent_len,
        'n_basis_points': len(fps_indices)
    }


if __name__ == "__main__":
    results = mnist_manifold_learning_example()
