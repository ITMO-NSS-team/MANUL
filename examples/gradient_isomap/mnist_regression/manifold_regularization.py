import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import datasets

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from regularizator.GraphRegTrainer import GraphRegTrainer
from utils.utils import split_data
import warnings

warnings.filterwarnings('ignore', category=FutureWarning,
                        message='unique with argument that is not not a Series')


def calc_accuracy(targets, predictions):
    """
    Calculate accuracy for regression by rounding predictions to nearest integer.
    """
    predictions_rounded = np.round(predictions).astype(int)
    targets_int = targets.astype(int)
    correct = np.sum(predictions_rounded == targets_int)
    return correct / len(targets_int)


def manifold_regularization(folder_path, model, num_epochs, batch_size, learning_rate, early_stop_patience,
                            lambda_method, adaptive_lambda_recompute=False):
    print(f"\n{'=' * 60}")
    print("STAGE 2: GRAPH REGULARIZATION TRAINING")
    print(f"{'=' * 60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(folder_path, f'regularization_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    print(f"✓ Experiment folder: {experiment_folder}")
    print(f"✓ Device: {device}")

    print("\n📂 Loading data...")
    try:
        n_samples = int(folder_path.split('/')[-2].split('_')[-1])
        mnist_dataset = datasets.MNIST(root='../data', train=True, download=True)
        X = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
        y = mnist_dataset.targets.numpy()
        X = X[:n_samples]
        y = y[:n_samples]
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # fps_indices.npy is cached one level up (in the outputs_dir shared across
        # runs) by manifold_learning.py, while best_distance_matrix.npy is written
        # into this specific run's folder.
        mnist_folder = folder_path.split('/')[-2]
        fps_indices = np.load(f'{mnist_folder}/fps_indices.npy')
        manifold_dist_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')
    except Exception as e:
        print(f'Error loading data from {folder_path}: {e}')
        print('Required files:\n1) fps_indices.npy\n2) best_distance_matrix.npy')
        print("\n❌ Please run manifold learning first!")
        raise

    print(f"\n{'=' * 40}")
    print(f"TRAINING REGULARIZED MODEL")
    print(f"{'=' * 40}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = GraphRegTrainer(
        train_features=X_train,
        train_target=y_train,
        val_features=X_val,
        val_targets=y_val,
        weights_matrix=manifold_dist_matrix,
        base_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        cache_folder=experiment_folder
    )

    trainer.train(
        plot_convergence=True,
        adaptive_lambda=lambda_method,
        early_stopping_patience=early_stop_patience,
        adaptive_lambda_recompute=adaptive_lambda_recompute,
    )

    fl64 = torch.float64

    X_train_t = torch.tensor(X_train, dtype=fl64).to(device)
    reg_pred_train = trainer.best_model(X_train_t).flatten().cpu().detach().numpy()
    reg_train_mse = mean_squared_error(y_train, reg_pred_train)
    reg_train_mae = mean_absolute_error(y_train, reg_pred_train)
    reg_train_r2 = r2_score(y_train, reg_pred_train)
    reg_train_accuracy = calc_accuracy(y_train, reg_pred_train)

    X_val_t = torch.tensor(X_val, dtype=fl64).to(device)
    reg_pred_val = trainer.best_model(X_val_t).flatten().cpu().detach().numpy()
    reg_val_mse = mean_squared_error(y_val, reg_pred_val)
    reg_val_mae = mean_absolute_error(y_val, reg_pred_val)
    reg_val_r2 = r2_score(y_val, reg_pred_val)
    reg_val_accuracy = calc_accuracy(y_val, reg_pred_val)

    X_test_t = torch.tensor(X_test, dtype=fl64).to(device)
    reg_pred_test = trainer.best_model(X_test_t).flatten().cpu().detach().numpy()
    reg_test_mse = mean_squared_error(y_test, reg_pred_test)
    reg_test_mae = mean_absolute_error(y_test, reg_pred_test)
    reg_test_r2 = r2_score(y_test, reg_pred_test)
    reg_test_accuracy = calc_accuracy(y_test, reg_pred_test)

    print(f"✓ Regularized - Train MSE: {reg_train_mse:.6f}, MAE: {reg_train_mae:.6f}, "
          f"R²: {reg_train_r2:.6f}, accuracy: {reg_train_accuracy}")
    print(f"✓ Regularized - Val MSE: {reg_val_mse:.6f}, MAE: {reg_val_mae:.6f}, "
          f"R²: {reg_val_r2:.6f}, accuracy: {reg_val_accuracy}")
    print(f"✓ Regularized - Test MSE: {reg_test_mse:.6f}, MAE: {reg_test_mae:.6f}, "
          f"R²: {reg_test_r2:.6f}, accuracy: {reg_test_accuracy}")

    metrics_df = pd.DataFrame([{
        'train_mse': reg_train_mse,
        'train_mae': reg_train_mae,
        'train_r2': reg_train_r2,
        'val_mse': reg_val_mse,
        'val_mae': reg_val_mae,
        'val_r2': reg_val_r2,
        'test_mse': reg_test_mse,
        'test_mae': reg_test_mae,
        'test_r2': reg_test_r2
    }])
    metrics_df.to_csv(os.path.join(experiment_folder, 'metrics.csv'), index=False)
    return experiment_folder


if __name__ == "__main__":
    folder_path = 'outputs_60000/mnist_run_20260115_234826'
    num_epochs = 1000
    batch_size = 1000
    lr = 0.01
    early_stop_patience = 100
    lambda_method = None

    model_architecture = nn.Sequential(
        nn.Linear(784, 1000, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(1000, 500, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(500, 32, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(32, 1, dtype=torch.float64),
        nn.ReLU(),
    )

    manifold_regularization(folder_path, model_architecture, num_epochs, batch_size, lr, early_stop_patience,
                            lambda_method)
