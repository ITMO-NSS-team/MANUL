import os.path

import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from Adam.GradientIsomap import GradientIsomap
from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector


def mnist_manifold_learning_example():
    """
    MNIST manifold learning example with FPS sampling and local PCA dimension estimation.
    """
    # Number of basis points (FPS samples)
    n_samples = 500
    latent_len = 300  # precomp latent dimension for Isomap

    #working_folder = f'mnist_{datetime.now().strftime("%d%m%Y-%H.%M")}'
    working_folder = os.path.join('mnist_isomap', f'mnist_{n_samples}')
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    # Load MNIST data
    print("Loading MNIST data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Convert to numpy and flatten images
    X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1)  # (60000, 784)
    y = mnist_dataset.targets.numpy()

    # Split train/test (80/20)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


    # print("\n=== DIMENSIONALITY ANALYSIS ===")
    # analyser = DimensionalityAnalyser()
    #
    # analyser.analyse_dimensions(
    #     X_train,
    #     method='both',
    #     n_samples=1000
    # )
    # analyser.plot_dimension_histograms(dataset_name="MNIST",
    #                                                  save_path=f'{working_folder}/hist_plot.png')
    # _, _ = analyser.plot_variance_threshold_analysis(X_train,
    #                                                  dataset_name="MNIST",
    #                                                  n_samples=1000,
    #                                                  save_path=f'{working_folder}/variance_plot.png')
    # latent_len = analyser.get_latent_dim(method='eigenvalue')

    # Apply FPS to extract 2000 key points from training data
    if not os.path.exists(f'{working_folder}/fps_indices.npy'):
        print("Applying Farthest Point Sampling...")
        fps_indices = memory_efficient_fps(X_train, n_samples)
        np.save(f'{working_folder}/fps_indices.npy', fps_indices)
        print(f"Sampled {len(fps_indices)} key points from training data")
    else:
        fps_indices = np.load(f'{working_folder}/fps_indices.npy')

    X_train_sparse = X_train[fps_indices]
    y_train_sparse = y_train[fps_indices]

    # Run manifold learning with GradientIsomap
    print("Starting manifold learning...")

    # Convert to torch tensors for GradientIsomap
    train_features = torch.tensor(X_train_sparse, dtype=torch.float32).to('cuda')
    train_target = torch.tensor(y_train_sparse, dtype=torch.float32).to('cuda')

    isomap = GradientIsomap(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_len,
        checkpoint_each=100,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=20
    )

    isomap.train()
    isomap.visualize_trained()

    # Get results - trained distance matrix
    best_distances_matrix = isomap.best_distances_matrix
    proj_features = isomap.best_isomap_model()

    # Get projections from Isomap - these are already computed!
    base_projections = proj_features.detach().cpu().numpy()

    # Reconstruct full distance matrix from upper triangular form
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

    # Create Projector and compute all projections
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

    # Compute projections for all training points
    projector.compute_all_projections()

    # Save data for graph regularization training
    print("\n=== SAVING DATA FOR GRAPH REGULARIZATION ===")
    np.save(f'{working_folder}/X_train.npy', X_train)
    np.save(f'{working_folder}/X_test.npy', X_test)
    np.save(f'{working_folder}/y_train.npy', y_train)
    np.save(f'{working_folder}/y_test.npy', y_test)
    np.save(f'{working_folder}/latent_dim.npy', latent_len)

    # Save projections using Projector
    projector.save(working_folder)

    print(f"\nAll data saved to {working_folder}/")

    return {
        'working_folder': working_folder,
        'latent_dim': latent_len,
        'n_basis_points': len(fps_indices)
    }


results = mnist_manifold_learning_example()