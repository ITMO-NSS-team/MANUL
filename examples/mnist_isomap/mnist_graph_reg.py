import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from regularizator.GraphRegTrainer import GraphRegTrainer


def load_data_from_folder(folder_path):
    """
    Load all necessary data from the folder created by mnist_isomap.py

    Args:
        folder_path: path to the folder with saved data

    Returns:
        dict with all loaded data
    """
    print(f"Loading data from {folder_path}...")

    # Load training and test data
    X_train = np.load(f'{folder_path}/X_train.npy')
    X_test = np.load(f'{folder_path}/X_test.npy')
    y_train = np.load(f'{folder_path}/y_train.npy')
    y_test = np.load(f'{folder_path}/y_test.npy')

    # Load distance matrix (upper triangular form)
    best_distances_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')

    # Load FPS indices (basis points)
    fps_indices = np.load(f'{folder_path}/fps_indices.npy')

    # Load latent dimension
    latent_dim = int(np.load(f'{folder_path}/latent_dim.npy'))

    # Load precomputed base projections (from GradientIsomap)
    base_projections = np.load(f'{folder_path}/base_projections.npy')

    # Load precomputed all projections (from Projector)
    all_projections = np.load(f'{folder_path}/all_projections.npy')

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  FPS indices: {len(fps_indices)} basis points")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Distance matrix: {best_distances_matrix.shape}")
    print(f"  Base projections: {base_projections.shape} precomputed")
    print(f"  All projections: {all_projections.shape} precomputed")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'best_distances_matrix': best_distances_matrix,
        'fps_indices': fps_indices,
        'latent_dim': latent_dim,
        'base_projections': base_projections,
        'all_projections': all_projections,
        'folder_path': folder_path
    }


def reconstruct_distance_matrix(best_distances_matrix, n_basis):
    """
    Reconstruct full symmetric distance matrix from upper triangular form

    Args:
        best_distances_matrix: upper triangular distances (1D array)
        n_basis: number of basis points

    Returns:
        weights_matrix: full symmetric distance matrix [n_basis, n_basis]
    """
    print("\nReconstructing full distance matrix...")
    weights_matrix = np.zeros((n_basis, n_basis))
    idx = 0
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            weights_matrix[i, j] = best_distances_matrix[idx]
            weights_matrix[j, i] = best_distances_matrix[idx]
            idx += 1

    print(f"  Reconstructed matrix shape: {weights_matrix.shape}")
    return weights_matrix


class MNISTClassifier(nn.Module):
    """
    Neural network classifier for MNIST
    """
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256, dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128, dtype=torch.float64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes, dtype=torch.float64)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x


def train_and_evaluate_model(X_train, y_train, X_test, y_test,
                             weights_matrix, fps_indices, base_projections, all_projections,
                             lambda_graph, model_name,
                             cache_folder, adaptive_lambda=False):
    """
    Train and evaluate a model with specified graph regularization

    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        weights_matrix: distance matrix
        fps_indices: basis point indices
        base_projections: precomputed projections from GradientIsomap
        all_projections: precomputed projections of all points from Projector
        lambda_graph: graph regularization coefficient (0 = no regularization)
        model_name: name for logging
        cache_folder: folder to save results
        adaptive_lambda: whether to use adaptive lambda

    Returns:
        dict with accuracy and predictions
    """
    print(f"\n--- Training {model_name} (lambda_graph={lambda_graph}) ---")

    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = GraphRegTrainer(
        train_features=X_train,
        train_target=y_train,
        weights_matrix=weights_matrix,
        basis_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        batch_size=128,
        lambda_graph=lambda_graph,
        n_neighbors=10,
        method='ensemble_knn',
        cache_folder=cache_folder,
        verbose=True,
        precomputed_base_projections=base_projections,
        precomputed_all_projections=all_projections
    )

    trainer.train(plot_convergence=True, adaptive_lambda=adaptive_lambda)
    trainer.save_weights()

    # Evaluate on test set
    pred = trainer.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_test, pred_classes)

    print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, pred_classes, digits=4))

    return {
        'accuracy': accuracy,
        'predictions': pred_classes,
        'trainer': trainer
    }


def mnist_graph_regularization(folder_path='mnist_test2'):
    """
    Main function to train MNIST classifier with graph regularization

    Args:
        folder_path: path to folder with data from mnist_isomap.py
    """
    # Load all data
    data = load_data_from_folder(folder_path)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    best_distances_matrix = data['best_distances_matrix']
    fps_indices = data['fps_indices']
    latent_dim = data['latent_dim']
    base_projections = data['base_projections']  # Готовые проекции из GradientIsomap
    all_projections = data['all_projections']    # Готовые проекции всех точек из Projector

    # Reconstruct full distance matrix
    n_basis = len(fps_indices)
    weights_matrix = reconstruct_distance_matrix(best_distances_matrix, n_basis)

    print("\n" + "="*60)
    print("TRAINING CLASSIFIER WITH GRAPH REGULARIZATION")
    print("="*60)
    print(f"✓ Using precomputed projections - skipping Isomap AND KNN computation!")

    # Train baseline model (no graph regularization)
    baseline_results = train_and_evaluate_model(
        X_train, y_train, X_test, y_test,
        weights_matrix, fps_indices, base_projections, all_projections,
        lambda_graph=0.0,
        model_name="BASELINE",
        cache_folder=f'{folder_path}/baseline',
        adaptive_lambda=False
    )

    # Train regularized model (with graph regularization)
    reg_results = train_and_evaluate_model(
        X_train, y_train, X_test, y_test,
        weights_matrix, fps_indices, base_projections, all_projections,
        lambda_graph=0.0001,
        model_name="REGULARIZED",
        cache_folder=f'{folder_path}/regularized',
        adaptive_lambda=False
    )

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    baseline_acc = baseline_results['accuracy']
    reg_acc = reg_results['accuracy']


    print(f"Baseline Accuracy:    {baseline_acc:.4f}")
    print(f"Regularized Accuracy: {reg_acc:.4f}")


    # Save comparison results
    comparison_results = {
        'baseline_accuracy': baseline_acc,
        'regularized_accuracy': reg_acc,
        'latent_dim': latent_dim,
        'n_basis_points': n_basis
    }

    np.save(f'{folder_path}/comparison_results.npy', comparison_results)
    print(f"\nComparison results saved to {folder_path}/comparison_results.npy")

    return comparison_results



folder_path = os.path.join('mnist_isomap', 'mnist_500')


results = mnist_graph_regularization(folder_path)