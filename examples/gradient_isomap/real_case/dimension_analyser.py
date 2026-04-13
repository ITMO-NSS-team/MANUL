import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps
from dataset_loader import load_dataset


def plot_3d_comparison(features_original, features_sparse, target_original, target_sparse, method='pca'):
    """3D comparison with dimensionality reduction"""
    print(f"  Reducing dimensions using {method.upper()}...")

    n_sample_original = min(5000, len(features_original))
    sample_indices = np.random.choice(len(features_original), n_sample_original, replace=False)
    features_original_sample = features_original[sample_indices].reshape(n_sample_original, -1)
    target_original_sample = target_original[sample_indices]

    if method == 'pca':
        reducer_cls = lambda: PCA(n_components=3, random_state=42)
    elif method == 'tsne':
        reducer_cls = lambda: TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    features_original_3d = reducer_cls().fit_transform(features_original_sample)
    features_sparse_flat = features_sparse.reshape(features_sparse.shape[0], -1)
    features_sparse_3d = reducer_cls().fit_transform(features_sparse_flat)

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(features_original_3d[:, 0], features_original_3d[:, 1], features_original_3d[:, 2],
                c=target_original_sample, cmap='tab10', alpha=0.7, s=20)
    ax1.set_title(f'Original Data ({method.upper()})\n{n_sample_original} samples')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(features_sparse_3d[:, 0], features_sparse_3d[:, 1], features_sparse_3d[:, 2],
                c=target_sparse, cmap='tab10', alpha=0.8, s=40)
    ax2.set_title(f'Sparse Data (FPS + {method.upper()})\n{len(features_sparse)} samples')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def analyze_dataset(features, target, dataset_name, n_fps_samples=2000, save_prefix=""):
    """Complete analysis pipeline for a dataset using DimensionalityAnalyser from utils."""
    print(f"\nCalculating optimal intrinsic dimension for {dataset_name}")

    print("  Memory-efficient FPS...")
    pts = memory_efficient_fps(features, n_samples=n_fps_samples, batch_size=500)
    features_sparse = features[pts]
    target_sparse = target[pts]

    if save_prefix:
        np.save(f'data_analysis/{save_prefix}_pts_{n_fps_samples}', pts)

    plot_3d_comparison(features, features_sparse, target, target_sparse, method='pca')

    analyser = DimensionalityAnalyser(max_neighbors=1000)

    analyser.analyse_dimensions(features, n_samples=1000, method='both')

    analyser.plot_dimension_histograms(
        dataset_name=dataset_name,
        save_path=f'data_analysis/{save_prefix}_histogram.png' if save_prefix else None
    )

    _, _ = analyser.plot_variance_threshold_analysis(
        features,
        dataset_name=dataset_name,
        thresholds=[0.90, 0.95],
        n_samples=1000,
        save_path=f'data_analysis/{save_prefix}_explained_var.png' if save_prefix else None
    )

    if save_prefix:
        analyser.save_results(f'data_analysis/{save_prefix}_dim_stats.json')

    optimal_dim = analyser.get_latent_dim(method='eigenvalue')
    return analyser.results['eigenvalue'], optimal_dim


if __name__ == "__main__":
    os.makedirs('data_analysis', exist_ok=True)

    TRAIN_RATIO = 0.7
    N_FPS_SAMPLES = 2000
    DATASET_SIZES = {'mnist': 60000, 'fmnist': 60000, 'cifar10': 50000}
    optimal_dims = {}

    print("\n" + "=" * 60)
    print("MNIST")
    print("=" * 60)

    n_load = int(DATASET_SIZES['mnist'] * TRAIN_RATIO)
    X_mnist, y_mnist = load_dataset('mnist', n_samples=n_load, random_state=42)
    _, opt_dim = analyze_dataset(X_mnist, y_mnist, "MNIST",
                                 n_fps_samples=N_FPS_SAMPLES, save_prefix="mnist")
    optimal_dims['MNIST'] = opt_dim


    print("\n" + "=" * 60)
    print("FASHION-MNIST")
    print("=" * 60)

    n_load = int(DATASET_SIZES['fmnist'] * TRAIN_RATIO)
    X_fmnist, y_fmnist = load_dataset('fmnist', n_samples=n_load, random_state=42)
    _, opt_dim = analyze_dataset(X_fmnist, y_fmnist, "Fashion-MNIST",
                                 n_fps_samples=N_FPS_SAMPLES, save_prefix="fmnist")
    optimal_dims['Fashion-MNIST'] = opt_dim


    print("\n" + "=" * 60)
    print("CIFAR-10")
    print("=" * 60)

    n_load = int(DATASET_SIZES['cifar10'] * TRAIN_RATIO)
    X_cifar, y_cifar = load_dataset('cifar10', n_samples=n_load, random_state=42)
    _, opt_dim = analyze_dataset(X_cifar, y_cifar, "CIFAR-10",
                                 n_fps_samples=N_FPS_SAMPLES, save_prefix="cifar10")
    optimal_dims['CIFAR-10'] = opt_dim

    print("\n" + "#" * 50)
    print("Results (90th percentile):")
    for ds_name, dim in optimal_dims.items():
        print(f"  * {ds_name.ljust(15)}: {dim}")
    print("#" * 50)
    print(f"Saved to data_analysis/")