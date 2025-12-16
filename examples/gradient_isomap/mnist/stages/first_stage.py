import os
import sys
import json
from datetime import datetime
import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split

from Adam.GradientIsomap import GradientIsomap
from utils.DimensionalityAnalyser import DimensionalityAnalyser
from utils.fps_implementation import memory_efficient_fps
from utils.Projector import Projector
from utils.cache_utils import load_or_compute_fps, set_global_seed
from regularizator.GraphRegTrainer import project_ensemble_knn

RANDOM_SEED = 42
def mnist_manifold_learning_example(save_checkpoint_history=False):
    """
    MNIST manifold learning example with FPS sampling and local PCA dimension estimation.
    """

    set_global_seed(RANDOM_SEED)

    n_samples = 2000

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(script_dir, '..'))
    project_root = os.path.abspath(os.path.join(script_dir, '../../../..'))

    outputs_dir = os.path.join(experiment_dir, 'outputs')
    data_dir = os.path.join(project_root, 'data')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    working_folder = os.path.join(outputs_dir, f'run_{timestamp}_n{n_samples}')
    os.makedirs(working_folder, exist_ok=True)

    print(f"Working folder: {working_folder}")
    print("Loading MNIST data...")


    mnist_dataset = datasets.MNIST(root=data_dir, train=True, download=True)

    X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
    y = mnist_dataset.targets.numpy()

    print("Splitting data into train/val/test...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    print("\n=== DIMENSIONALITY ANALYSIS ===")
    analyser = DimensionalityAnalyser()

    analyser.analyse_dimensions(
        X_train,
        method='eigenvalue',
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

    print(f"Starting Isomap training for 15000 epochs...")
    isomap = GradientIsomap(
        train_feature=train_features,
        train_target=train_target,
        latent_len=latent_len,
        n_neighbors=25,
        checkpoint_each=100,
        save_checkpoint_history=save_checkpoint_history,
        logs_folder=working_folder,
        plot_convergence=False,
        epochs=15000,
        stop_criteria_value=0.001,
    )
    isomap.train()
    isomap.visualize_trained()

    best_distances_matrix = isomap.best_distances_matrix
    proj_features = isomap.best_isomap_model()
    base_projections = proj_features.detach().cpu().numpy()

    np.save(f'{working_folder}/base_projections.npy', base_projections)

    np.save(f'{working_folder}/best_distance_matrix.npy', best_distances_matrix)
    print(f"Saved distance matrix and base projections")
    print(f"Latent_dim: {latent_len}")

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
        batch_size=128,
        precomputed_base_projections=base_projections,
        verbose=True
    )
    projector.compute_projection()
    train_projections = projector.projection
    np.save(train_proj_path, train_projections)
    print(f"Saved train projections")

    experiment_metadata = {
        'dataset_type': 'mnist',
        'random_seed': RANDOM_SEED,
        'latent_dim': int(latent_len),
        'split_params': {
            'test_size_outer': 0.2,
            'test_size_inner': 0.2
        }
    }
    metadata_path = os.path.join(working_folder, 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    print(f"Saved experiment metadata to {metadata_path}")

    print(f"\nAll data saved to {working_folder}/")


if __name__ == "__main__":
    mnist_manifold_learning_example(save_checkpoint_history=False)
