import os.path

import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from Adam.GradientIsomap import GradientIsomap
from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps


def mnist_manifold_learning_example():
    """
    MNIST manifold learning example with FPS sampling and local PCA dimension estimation.
    """
    #working_folder = f'mnist_{datetime.now().strftime("%d%m%Y-%H.%M")}'
    working_folder = f'mnist_test2'
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

    # Apply FPS to extract 2000 key points from training data
    if not os.path.exists(f'{working_folder}/fps_indices.npy'):
        print("Applying Farthest Point Sampling...")
        fps_indices = memory_efficient_fps(X_train, 2000)
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
        epochs=200
    )

    isomap.train()
    isomap.visualize_trained()

    # Get results
    matrix = isomap.best_distances_matrix
    proj_features = isomap.best_isomap_model()


if __name__ == "__main__":
    results = mnist_manifold_learning_example()