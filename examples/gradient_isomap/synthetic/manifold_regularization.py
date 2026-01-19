import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import float64 as fl64

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from regularizator.GraphRegTrainer import GraphRegTrainer
from utils.utils import split_data
import warnings

warnings.filterwarnings('ignore', category=FutureWarning,
                        message='unique with argument that is not not a Series')


def manifold_regularization(folder_path, model, num_epochs, batch_size, learning_rate, early_stop_patience,
                            lambda_method):
    print(f"\n{'=' * 60}")
    print("STAGE 2: GRAPH REGULARIZATION TRAINING")
    print(f"{'=' * 60}\n")

    geometry_name = os.path.basename(folder_path).split('_')[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(folder_path, f'regularization_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    print(f"✓ Experiment folder: {experiment_folder}")
    print(f"✓ Geometry: {geometry_name}")

    print("\n📂 Loading data...")
    try:
        fps_indices = np.load(f'{folder_path}/fps_indices.npy')
        manifold_dist_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')
        features = np.load(f'{folder_path}/all_features.npy')
        targets = np.load(f'{folder_path}/all_targets.npy')
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets)
    except Exception as e:
        print(f'Folder {folder_path} does not have required files:\n 1) '
              f'fps_indices.npy\n2)best_distance_matrix.npy\n3)all_features.npy\n4)all_targets.npy')
        print("\n❌ Please run manifold learning or specify the correct folder!")
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
    )

    # Train set metrics
    X_train = torch.tensor(X_train, dtype=fl64).to('cuda')
    y_train = torch.tensor(y_train, dtype=fl64).to('cuda')
    reg_pred_train = trainer.best_model(X_train).flatten().cpu().detach().numpy()
    y_train = y_train.flatten().cpu().detach().numpy()
    reg_train_mse = mean_squared_error(y_train, reg_pred_train)
    reg_train_mae = mean_absolute_error(y_train, reg_pred_train)
    reg_train_r2 = r2_score(y_train, reg_pred_train)

    # Validation set metrics
    X_val = torch.tensor(X_val, dtype=fl64).to('cuda')
    y_val = torch.tensor(y_val, dtype=fl64).to('cuda')
    reg_pred_val = trainer.best_model(X_val).flatten().cpu().detach().numpy()
    y_val = y_val.flatten().cpu().detach().numpy()
    reg_val_mse = mean_squared_error(y_val, reg_pred_val)
    reg_val_mae = mean_absolute_error(y_val, reg_pred_val)
    reg_val_r2 = r2_score(y_val, reg_pred_val)

    # Test set metrics
    X_test = torch.tensor(X_test, dtype=fl64).to('cuda')
    y_test = torch.tensor(y_test, dtype=fl64).to('cuda')
    reg_pred_test = trainer.best_model(X_test).flatten().cpu().detach().numpy()
    y_test = y_test.flatten().cpu().detach().numpy()
    reg_test_mse = mean_squared_error(y_test, reg_pred_test)
    reg_test_mae = mean_absolute_error(y_test, reg_pred_test)
    reg_test_r2 = r2_score(y_test, reg_pred_test)

    print(f"✓ Regularized - Train MSE: {reg_train_mse:.6f}, MAE: {reg_train_mae:.6f}, R²: {reg_train_r2:.6f}")
    print(f"✓ Regularized - Val MSE: {reg_val_mse:.6f}, MAE: {reg_val_mae:.6f}, R²: {reg_val_r2:.6f}")
    print(f"✓ Regularized - Test MSE: {reg_test_mse:.6f}, MAE: {reg_test_mae:.6f}, R²: {reg_test_r2:.6f}")

    metrics_df = pd.DataFrame([{
        'geometry': geometry_name,
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
    folder_path = 'outputs_stat_0.01noise_5k_sobol_v3\sphere\sphere_run_20260114_201528'
    num_epochs = 1000
    batch_size = 2048
    lr = 1e-3
    early_stop_patience = 100
    lambda_method = None

    model_architecture = nn.Sequential(
        nn.Linear(3, 32, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(32, 1, dtype=torch.float64)
    )

    manifold_regularization(folder_path, model_architecture, num_epochs, batch_size, lr, early_stop_patience,
                            lambda_method)
