"""
Stage 2: Graph Regularization Training for Synthetic Geometries

This script trains regression models with graph regularization on synthetic geometries.
It follows the MNIST pipeline but adapted for regression tasks:
1. Load preprocessed data from Stage 1
2. Train baseline model (lambda_graph=0.0)
3. Train regularized model (lambda_graph=0.0001)
4. Compare results with comprehensive visualizations
5. Generate summary table across all geometries
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from regularizator.GraphRegTrainer import GraphRegTrainer
from utils.cache_utils import check_required_files

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_from_folder(folder_path):
    """
    Load all necessary data from the folder created by synthetic_geometry_manifold_learning.py

    Args:
        folder_path: path to the folder with saved data

    Returns:
        dict with all loaded data
    """
    print(f"\nLoading data from {folder_path}...")

    X_train = np.load(os.path.join(folder_path, 'X_train.npy'))
    X_val = np.load(os.path.join(folder_path, 'X_val.npy'))
    X_test = np.load(os.path.join(folder_path, 'X_test.npy'))
    y_train = np.load(os.path.join(folder_path, 'y_train.npy'))
    y_val = np.load(os.path.join(folder_path, 'y_val.npy'))
    y_test = np.load(os.path.join(folder_path, 'y_test.npy'))

    best_distances_matrix = np.load(os.path.join(folder_path, 'best_distance_matrix.npy'))
    fps_indices = np.load(os.path.join(folder_path, 'fps_indices.npy'))
    latent_dim = int(np.load(os.path.join(folder_path, 'latent_dim.npy')))
    base_projections = np.load(os.path.join(folder_path, 'base_projections.npy'))
    train_projections = np.load(os.path.join(folder_path, 'train_projections.npy'))
    val_projections = np.load(os.path.join(folder_path, 'val_projections.npy'))

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  FPS indices: {len(fps_indices)} basis points")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Base projections: {base_projections.shape}")
    print(f"  Train projections: {train_projections.shape}")
    print(f"  Val projections: {val_projections.shape}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'best_distances_matrix': best_distances_matrix,
        'fps_indices': fps_indices,
        'latent_dim': latent_dim,
        'base_projections': base_projections,
        'train_projections': train_projections,
        'val_projections': val_projections,
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
    print("  Reconstructing full distance matrix...")
    weights_matrix = np.zeros((n_basis, n_basis))
    idx = 0
    for i in range(n_basis):
        for j in range(i+1, n_basis):
            weights_matrix[i, j] = best_distances_matrix[idx]
            weights_matrix[j, i] = best_distances_matrix[idx]
            idx += 1

    print(f"    Matrix shape: {weights_matrix.shape}")
    return weights_matrix


class RegressionModel(nn.Module):
    """
    Simple neural network for regression task
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0], dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], output_dim, dtype=torch.float64)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test,
                             weights_matrix, fps_indices, base_projections,
                             train_projections, val_projections,
                             lambda_graph, model_name,
                             cache_folder, num_epochs, batch_size,
                             learning_rate, early_stopping_patience, adaptive_lambda=False):
    """
    Train and evaluate a regression model with specified graph regularization

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        weights_matrix: distance matrix
        fps_indices: basis point indices
        base_projections: precomputed projections from GradientIsomap
        train_projections: precomputed projections for training data
        val_projections: precomputed projections for validation data
        lambda_graph: graph regularization coefficient (0 = no regularization)
        model_name: name for logging
        cache_folder: folder to save results
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate
        early_stopping_patience: patience for early stopping

    Returns:
        dict with metrics, predictions, and trainer
    """
    print(f"\n  --- Training {model_name} (lambda_graph={lambda_graph}) ---")


    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    model = RegressionModel(input_dim=X_train.shape[1], hidden_dims=[64, 32], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = GraphRegTrainer(
        train_features=X_train,
        train_target=y_train,
        weights_matrix=weights_matrix,
        basis_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lambda_graph=lambda_graph,
        n_neighbors=10,
        method='ensemble_knn',
        cache_folder=cache_folder,
        verbose=True,
        precomputed_base_projections=base_projections,
        precomputed_all_projections=train_projections
    )

    trainer.train(
        plot_convergence=True,
        adaptive_lambda=adaptive_lambda,
        early_stopping_patience=early_stopping_patience,
        val_features=X_val,
        val_target=y_val,
        val_projections=val_projections
    )

    trainer.load_best_weights()

    # Evaluate on validation set
    pred_val = trainer.predict(X_val).flatten()
    y_val_flat = y_val.flatten()
    val_mse = mean_squared_error(y_val_flat, pred_val)
    val_mae = mean_absolute_error(y_val_flat, pred_val)
    val_r2 = r2_score(y_val_flat, pred_val)

    print(f"    {model_name} Validation - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")

    pred_test = trainer.predict(X_test).flatten()
    y_test_flat = y_test.flatten()
    test_mse = mean_squared_error(y_test_flat, pred_test)
    test_mae = mean_absolute_error(y_test_flat, pred_test)
    test_r2 = r2_score(y_test_flat, pred_test)

    print(f"    {model_name} Test - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")

    train_losses = trainer.trained_loss_values.get('model_loss', [])
    val_losses = trainer.trained_loss_values.get('val_loss', [])

    return {
        'val_mse': val_mse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions_val': pred_val,
        'predictions_test': pred_test,
        'trainer': trainer,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': trainer.best_epoch if hasattr(trainer, 'best_epoch') else num_epochs
    }


def create_comparison_visualization(geometry_name, X_test, y_test,
                                   baseline_results, reg_results, save_path):
    """
    Create comparison visualization with 2 subplots:
    1. Training loss curves (baseline vs regularized)
    2. Validation loss curves (baseline vs regularized)

    Args:
        geometry_name: name of the geometry
        X_test: test data (3D coordinates)
        y_test: ground truth targets
        baseline_results: results from baseline model
        reg_results: results from regularized model
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    y_test_flat = y_test.flatten()
    pred_baseline = baseline_results['predictions_test']
    pred_reg = reg_results['predictions_test']

    ax1 = axes[0]

    baseline_train_losses = baseline_results.get('train_losses', [])
    reg_train_losses = reg_results.get('train_losses', [])

    if len(baseline_train_losses) > 0 and len(reg_train_losses) > 0:
        ax1.plot(baseline_train_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
        ax1.plot(reg_train_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Training Loss', fontsize=11)
        ax1.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

    ax2 = axes[1]

    baseline_val_losses = baseline_results.get('val_losses', [])
    reg_val_losses = reg_results.get('val_losses', [])

    if len(baseline_val_losses) > 0 and len(reg_val_losses) > 0:
        ax2.plot(baseline_val_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
        ax2.plot(reg_val_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Loss', fontsize=11)
        ax2.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    # Overall title
    mse_base = baseline_results['test_mse']
    mse_reg = reg_results['test_mse']
    improvement = ((mse_base - mse_reg) / mse_base * 100)

    fig.suptitle(f'{geometry_name.upper()} - Baseline vs Regularized Comparison\n'
                f'Test MSE Improvement: {improvement:.2f}% (Baseline: {mse_base:.6f}, Regularized: {mse_reg:.6f})',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Visualization saved to {save_path}")


def save_experiment_config(experiment_folder, geometry_name, baseline_params, reg_params, results):
    """
    Save experiment configuration and results to JSON file

    Args:
        experiment_folder: folder to save config
        geometry_name: name of geometry
        baseline_params: dict with baseline model parameters
        reg_params: dict with regularized model parameters
        results: dict with experiment results
    """
    config = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'geometry': geometry_name,
            'architecture': '64-32-1 (regression)'
        },
        'baseline_model': baseline_params,
        'regularized_model': reg_params,
        'results': results
    }

    config_path = os.path.join(experiment_folder, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nExperiment config saved to {config_path}")


def process_geometry(geometry_name,
                    results_base_path='synthetic_isomap/results',
                    baseline_lambda=0.0,
                    reg_lambda=0.0001,
                    num_epochs=15000,
                    batch_size=256,
                    learning_rate=1e-3,
                    early_stopping_patience=20000):
    """
    Process a single geometry: train models and create visualizations

    Args:
        geometry_name: name of the geometry
        results_base_path: base path to results folders
        baseline_lambda: lambda for baseline model
        reg_lambda: lambda for regularized model
        num_epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        early_stopping_patience: early stopping patience

    Returns:
        dict with comparison metrics
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {geometry_name.upper()}")
    print(f"{'='*80}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_path = os.path.join(results_base_path, geometry_name)
    experiment_folder = os.path.join(folder_path, f'experiment_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    print(f"EXPERIMENT FOLDER: {experiment_folder}")

    data = load_data_from_folder(folder_path)

    X_train = data['X_train']
    X_val = data['X_val']

    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    fps_indices = data['fps_indices']
    base_projections = data['base_projections']
    train_projections = data['train_projections']
    val_projections = data['val_projections']
    best_distances_matrix = data['best_distances_matrix']
    latent_dim = data['latent_dim']

    n_basis = len(fps_indices)
    weights_matrix = reconstruct_distance_matrix(best_distances_matrix, n_basis)

    print("\n  Training baseline model (no regularization)...")
    baseline_results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_matrix, fps_indices, base_projections,
        train_projections, val_projections,
        lambda_graph=baseline_lambda,
        model_name="BASELINE",
        cache_folder=os.path.join(experiment_folder, 'baseline'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )

    print(f"\n  Training regularized model (lambda={reg_lambda})...")
    reg_results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_matrix, fps_indices, base_projections,
        train_projections, val_projections,
        lambda_graph=reg_lambda,
        model_name="REGULARIZED",
        cache_folder=os.path.join(experiment_folder, 'regularized'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )

    print("\n  Creating comparison visualization...")
    viz_path = os.path.join(experiment_folder, f'{geometry_name}_comparison.png')
    create_comparison_visualization(
        geometry_name, X_test, y_test,
        baseline_results, reg_results, viz_path
    )

    baseline_params = {
        'lambda_graph': baseline_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'best_epoch': baseline_results.get('best_epoch', num_epochs)
    }

    reg_params = {
        'lambda_graph': reg_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'best_epoch': reg_results.get('best_epoch', num_epochs)
    }

    results = {
        'baseline_test_mse': float(baseline_results['test_mse']),
        'baseline_test_mae': float(baseline_results['test_mae']),
        'baseline_test_r2': float(baseline_results['test_r2']),
        'baseline_val_mse': float(baseline_results['val_mse']),
        'regularized_test_mse': float(reg_results['test_mse']),
        'regularized_test_mae': float(reg_results['test_mae']),
        'regularized_test_r2': float(reg_results['test_r2']),
        'regularized_val_mse': float(reg_results['val_mse']),
        'mse_improvement_percent': float(((baseline_results['test_mse'] - reg_results['test_mse']) /
                                          baseline_results['test_mse'] * 100)),
        'latent_dim': int(latent_dim),
        'n_basis_points': int(n_basis),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }

    save_experiment_config(experiment_folder, geometry_name, baseline_params, reg_params, results)

    comparison_metrics = {
        'geometry': geometry_name,
        'baseline_test_mse': baseline_results['test_mse'],
        'baseline_test_mae': baseline_results['test_mae'],
        'baseline_test_r2': baseline_results['test_r2'],
        'regularized_test_mse': reg_results['test_mse'],
        'regularized_test_mae': reg_results['test_mae'],
        'regularized_test_r2': reg_results['test_r2'],
        'mse_improvement_percent': ((baseline_results['test_mse'] - reg_results['test_mse']) /
                                   baseline_results['test_mse'] * 100),
    }

    np.save(os.path.join(experiment_folder, 'comparison_metrics.npy'), comparison_metrics)
    print(f" Metrics saved to {experiment_folder}/comparison_metrics.npy")

    return comparison_metrics


def create_summary_table(all_metrics, save_path):
    """
    Create and save summary table for all geometries

    Args:
        all_metrics: list of comparison metrics dicts
        save_path: path to save the summary
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE - ALL GEOMETRIES")
    print("="*80)

    header = f"{'Geometry':<20} {'Baseline MSE':<15} {'Regularized MSE':<15} {'Improvement':<12} {'R² (Base)':<12} {'R² (Reg)':<12}"
    print(header)
    print("-" * len(header))

    summary_lines = [header, "-" * len(header)]

    for metrics in all_metrics:
        line = (f"{metrics['geometry']:<20} "
               f"{metrics['baseline_test_mse']:<15.6f} "
               f"{metrics['regularized_test_mse']:<15.6f} "
               f"{metrics['mse_improvement_percent']:>10.2f}% "
               f"{metrics['baseline_test_r2']:>11.4f} "
               f"{metrics['regularized_test_r2']:>11.4f}")
        print(line)
        summary_lines.append(line)

    with open(save_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n✓ Summary table saved to {save_path}")




def synthetic_graph_regularization(folder_path=None,
                                    baseline_lambda=0.0,
                                    reg_lambda=0.00001,
                                    num_epochs=20000,
                                    batch_size=1024,
                                    learning_rate=1e-3,
                                    early_stopping_patience=20000,
                                    adaptive_lambda=True):
    """
    Main function to train regression model with graph regularization for synthetic geometries

    Args:
        folder_path: path to folder with data from Stage 1
        baseline_lambda: lambda for baseline model
        reg_lambda: lambda for regularized model
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        early_stopping_patience: patience for early stopping
    """
    required_files = [
        'fps_indices.npy',
        'best_distance_matrix.npy',
        'X_train.npy',
        'y_train.npy',
        'train_projections.npy',
        'val_projections.npy',
        'base_projections.npy',
        'latent_dim.npy'
    ]

    if not check_required_files(folder_path, required_files):
        print("\nPlease run Stage 1 (first_stage.py) first")
        return None

    geometry_name = os.path.basename(folder_path).split('_')[-1]  # e.g., run_20250106_143022_torus -> torus

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_folder = os.path.join(folder_path, f'experiment_{timestamp}')
    os.makedirs(experiment_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT FOLDER: {experiment_folder}")
    print(f"{'='*60}\n")

    data = load_data_from_folder(folder_path)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    best_distances_matrix = data['best_distances_matrix']
    fps_indices = data['fps_indices']
    latent_dim = data['latent_dim']
    base_projections = data['base_projections']
    train_projections = data['train_projections']
    val_projections = data['val_projections']

    n_basis = len(fps_indices)
    weights_matrix = reconstruct_distance_matrix(best_distances_matrix, n_basis)

    print("\n" + "="*60)
    print("TRAINING REGRESSOR WITH GRAPH REGULARIZATION")
    print("="*60)

    baseline_results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_matrix, fps_indices, base_projections,
        train_projections, val_projections,
        lambda_graph=baseline_lambda,
        model_name="Baseline",
        cache_folder=os.path.join(experiment_folder, 'baseline'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        adaptive_lambda=False
    )

    reg_results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_matrix, fps_indices, base_projections,
        train_projections, val_projections,
        lambda_graph=reg_lambda,
        model_name="REGULARIZED",
        cache_folder=os.path.join(experiment_folder, 'regularized'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        adaptive_lambda=adaptive_lambda
    )

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline Test MSE:    {baseline_results['test_mse']:.6f}")
    print(f"Regularized Test MSE: {reg_results['test_mse']:.6f}")

    print("\n  Creating comparison visualization...")
    viz_path = os.path.join(experiment_folder, f'{geometry_name}_normalized_n^2_comparison.png')
    create_comparison_visualization(
        geometry_name, X_test, y_test,
        baseline_results, reg_results, viz_path
    )

    baseline_params = {
        'lambda_graph': baseline_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'best_epoch': baseline_results.get('best_epoch', num_epochs)
    }

    reg_params = {
        'lambda_graph': reg_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'best_epoch': reg_results.get('best_epoch', num_epochs),
        'adaptive lambda': adaptive_lambda
    }

    results = {
        'baseline_test_mse': float(baseline_results['test_mse']),
        'baseline_val_mse': float(baseline_results['val_mse']),
        'regularized_test_mse': float(reg_results['test_mse']),
        'regularized_val_mse': float(reg_results['val_mse']),
        'mse_improvement_percent': float((baseline_results['test_mse'] - reg_results['test_mse']) /
                                        baseline_results['test_mse'] * 100),
        'latent_dim': int(latent_dim),
        'n_basis_points': int(n_basis),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }

    save_experiment_config(experiment_folder, geometry_name, baseline_params, reg_params, results)

    np.save(os.path.join(experiment_folder, 'comparison_results.npy'), results)
    print(f"Comparison results saved to {experiment_folder}/comparison_results.npy")

    return results


if __name__ == "__main__":
    # Specify the run folder to use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Set your run folder name
    run_folder_name = 'torus'
    folder_path = os.path.join(experiment_dir, 'outputs', run_folder_name)

    print(f"Script location: {script_dir}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Looking for data in: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"\nError: Folder not found: {folder_path}")
        print("\nPlease update 'run_folder_name' in this script or run first_stage.py first!")
        sys.exit(1)

    results = synthetic_graph_regularization(
        folder_path=folder_path,
        baseline_lambda=0.0,
        reg_lambda=1e-2,
        num_epochs=15000,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping_patience=150,
        adaptive_lambda=True
    )
