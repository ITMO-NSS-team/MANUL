import os
import sys
import json
from datetime import datetime
from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    Load all necessary data from the folder created by Stage 1

    Args:
        folder_path: path to the folder with saved data

    Returns:
        dict with all loaded data
    """
    print(f"Loading data from {folder_path}...")

    X_train = np.load(f'{folder_path}/X_train.npy')
    X_val = np.load(f'{folder_path}/X_val.npy')
    X_test = np.load(f'{folder_path}/X_test.npy')
    y_train = np.load(f'{folder_path}/y_train.npy')
    y_val = np.load(f'{folder_path}/y_val.npy')
    y_test = np.load(f'{folder_path}/y_test.npy')

    best_distances_matrix = np.load(f'{folder_path}/best_distance_matrix.npy')
    fps_indices = np.load(f'{folder_path}/fps_indices.npy')
    latent_dim = int(np.load(f'{folder_path}/latent_dim.npy'))
    base_projections = np.load(f'{folder_path}/base_projections.npy')
    train_projections = np.load(f'{folder_path}/train_projections.npy')

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  FPS indices: {len(fps_indices)} basis points")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Train projections: {train_projections.shape} precomputed")

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
        'folder_path': folder_path
    }


def reconstruct_distance_matrix(best_distances_matrix, n_basis):
    """
    Reconstruct full symmetric distance matrix from upper triangular form
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

    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64, dtype=torch.float64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes, dtype=torch.float64)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test,
                             weights_matrix, fps_indices, base_projections,
                             train_projections,
                             lambda_graph, model_name,
                             cache_folder, num_epochs, batch_size,
                             learning_rate, early_stopping_patience, adaptive_lambda=False):
    """
    Train and evaluate a model with specified graph regularization
    """
    print(f"\n--- Training {model_name} (lambda_graph={lambda_graph}) ---")
    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")


    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
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
        val_target=y_val
    )

    trainer.load_best_weights()

    pred_val = trainer.predict(X_val)
    pred_val_classes = np.argmax(pred_val, axis=1)
    val_accuracy = accuracy_score(y_val, pred_val_classes)
    print(f"\n{model_name} Validation Accuracy: {val_accuracy:.4f}")

    pred = trainer.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_test, pred_classes)

    print(f"\n{model_name} Test Accuracy: {accuracy:.4f}")
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, pred_classes, digits=4))

    train_losses = trainer.trained_loss_values.get('model_loss', [])
    val_losses = trainer.trained_loss_values.get('val_loss', [])
    train_accuracies = trainer.trained_loss_values.get('train_accuracy', [])
    val_accuracies = trainer.trained_loss_values.get('val_accuracy', [])

    return {
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'predictions': pred_classes,
        'trainer': trainer,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_epoch': trainer.best_epoch if hasattr(trainer, 'best_epoch') else num_epochs
    }


def create_mnist_comparison_visualization(baseline_results, reg_results, save_path):
    """
    Create comparison visualization for MNIST with 4 subplots (losses and accuracies)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    baseline_train_losses = baseline_results.get('train_losses', [])
    reg_train_losses = reg_results.get('train_losses', [])
    baseline_val_losses = baseline_results.get('val_losses', [])
    reg_val_losses = reg_results.get('val_losses', [])

    baseline_train_accs = baseline_results.get('train_accuracies', [])
    reg_train_accs = reg_results.get('train_accuracies', [])
    baseline_val_accs = baseline_results.get('val_accuracies', [])
    reg_val_accs = reg_results.get('val_accuracies', [])

    baseline_test_acc = baseline_results['accuracy']
    reg_test_acc = reg_results['accuracy']
    baseline_val_acc = baseline_results['val_accuracy']
    reg_val_acc = reg_results['val_accuracy']

    # Top-left: Training Loss
    ax1 = axes[0, 0]
    if len(baseline_train_losses) > 0 and len(reg_train_losses) > 0:
        ax1.plot(baseline_train_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
        ax1.plot(reg_train_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Training Loss', fontsize=11)
        ax1.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')


    ax2 = axes[0, 1]
    if len(baseline_val_losses) > 0 and len(reg_val_losses) > 0:
        ax2.plot(baseline_val_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
        ax2.plot(reg_val_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Loss', fontsize=11)
        ax2.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    ax3 = axes[1, 0]
    if len(baseline_train_accs) > 0 and len(reg_train_accs) > 0:
        # Create x-axis based on accuracy check interval (every 10 epochs)
        accuracy_check_interval = max(1, len(baseline_train_losses) // len(baseline_train_accs))
        epochs_acc = [i * accuracy_check_interval for i in range(len(baseline_train_accs))]

        ax3.plot(epochs_acc, baseline_train_accs, label='Baseline', linewidth=2, alpha=0.8, color='blue', marker='o', markersize=3)
        ax3.plot(epochs_acc, reg_train_accs, label='Regularized', linewidth=2, alpha=0.8, color='red', marker='s', markersize=3)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Training Accuracy', fontsize=11)
        ax3.set_title('Training Accuracy Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])

    ax4 = axes[1, 1]
    if len(baseline_val_accs) > 0 and len(reg_val_accs) > 0:
        accuracy_check_interval = max(1, len(baseline_val_losses) // len(baseline_val_accs))
        epochs_acc = [i * accuracy_check_interval for i in range(len(baseline_val_accs))]

        ax4.plot(epochs_acc, baseline_val_accs, label='Baseline', linewidth=2, alpha=0.8, color='blue', marker='o', markersize=3)
        ax4.plot(epochs_acc, reg_val_accs, label='Regularized', linewidth=2, alpha=0.8, color='red', marker='s', markersize=3)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Validation Accuracy', fontsize=11)
        ax4.set_title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1.05])

        ax4.text(0.02, 0.98, f'Final Val Acc:\nBaseline: {baseline_val_acc:.4f}\nRegularized: {reg_val_acc:.4f}',
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    improvement = (reg_test_acc - baseline_test_acc) * 100
    fig.suptitle(f'MNIST Classification: Baseline vs Regularized Comparison\n' +
                f'Test Accuracy: Baseline: {baseline_test_acc:.4f}, Regularized: {reg_test_acc:.4f} (Improvement: {improvement:+.2f}%)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison visualization saved to {save_path}")


def save_experiment_config(experiment_folder, baseline_params, reg_params, results):
    """
    Save experiment configuration and results to JSON file
    """
    config = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'MNIST',
            'architecture': '128-64-10 (simplified)'
        },
        'baseline_model': baseline_params,
        'regularized_model': reg_params,
        'results': results
    }

    config_path = os.path.join(experiment_folder, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nExperiment config saved to {config_path}")


def mnist_graph_regularization(folder_path: Optional[str] = None,
                               baseline_lambda: float = 0.0,
                               reg_lambda: float = 0.00001,
                               num_epochs: int = 4000,
                               batch_size: int = 128,
                               learning_rate: float = 1e-6,
                               early_stopping_patience: int = 10000,
                               adaptive_lambda: Union[bool, str] = False):
    """
    Main function to train MNIST classifier with graph regularization

    Args:
        folder_path: path to folder with data from Stage 1
        baseline_lambda: lambda for baseline model
        reg_lambda: lambda for regularized model
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        early_stopping_patience: patience for early stopping
        adaptive_lambda: 'sobol', 'gradnorm', False
    """
    required_files = [
        'fps_indices.npy',
        'best_distance_matrix.npy',
        'X_train.npy',
        'y_train.npy',
        'train_projections.npy',
        'base_projections.npy',
        'latent_dim.npy'
    ]

    if folder_path is None or not check_required_files(folder_path, required_files):
        print("\nPlease run Stage 1 (first_stage.py) first")
        return None

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


    n_basis = len(fps_indices)
    weights_matrix = reconstruct_distance_matrix(best_distances_matrix, n_basis)

    print("\n" + "="*60)
    print("TRAINING CLASSIFIER WITH GRAPH REGULARIZATION")
    print("="*60)
    print(f"Using precomputed projections")

    baseline_results = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        weights_matrix, fps_indices, base_projections,
        train_projections,
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
        train_projections,
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
    baseline_acc = baseline_results['accuracy']
    reg_acc = reg_results['accuracy']

    print(f"Baseline Accuracy:    {baseline_acc:.4f}")
    print(f"Regularized Accuracy: {reg_acc:.4f}")

    print("\n  Creating comparison visualization...")
    viz_path = os.path.join(experiment_folder, 'mnist_comparison.png')
    create_mnist_comparison_visualization(
        baseline_results, reg_results, viz_path
    )

    baseline_params = {
        'lambda_graph': baseline_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'adaptive_lambda': {'method': 'disabled'},
        'accuracy_check_interval': 10,
        'best_epoch': baseline_results.get('best_epoch', num_epochs)
    }

    # Determine adaptive lambda config
    adaptive_lambda_config = {}
    if adaptive_lambda == 'sobol':
        adaptive_lambda_config = {
            'method': 'sobol',
            'n_samples': 5,
            'sampling_D': 2,
            'warmup_fraction': 0.1
        }
    elif adaptive_lambda == 'gradnorm':
        adaptive_lambda_config = {
            'method': 'gradnorm',
            'alpha': 0.0001,
            'lr_weights': 0.0001,
            'initial_lambda_graph': reg_lambda
        }
    else:
        adaptive_lambda_config = {
            'method': 'disabled'
        }

    reg_params = {
        'lambda_graph': reg_lambda,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'adaptive_lambda': adaptive_lambda_config,
        'accuracy_check_interval': 10,
        'best_epoch': reg_results.get('best_epoch', num_epochs)
    }

    results = {
        'baseline_test_accuracy': float(baseline_acc),
        'baseline_val_accuracy': float(baseline_results['val_accuracy']),
        'regularized_test_accuracy': float(reg_acc),
        'regularized_val_accuracy': float(reg_results['val_accuracy']),
        'accuracy_improvement_percent': float((reg_acc - baseline_acc) * 100),
        'latent_dim': int(latent_dim),
        'n_basis_points': int(n_basis),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }

    save_experiment_config(experiment_folder, baseline_params, reg_params, results)
    pd.DataFrame([results]).to_csv(os.path.join(experiment_folder, 'comparison_results.csv'), index=False)
    print(f"Comparison results saved to {experiment_folder}/comparison_results.csv")

    return results


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(script_dir, '..'))  # mnist/

    # Set your run folder name
    run_folder_name = 'mnist_2000'
    folder_path = os.path.join(experiment_dir, 'outputs', run_folder_name)

    print(f"Script location: {script_dir}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Looking for data in: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"\nError: Folder not found: {folder_path}")
        print("\nPlease update 'run_folder_name' in this script or run first_stage.py")
        sys.exit(1)

    results = mnist_graph_regularization(
        folder_path=folder_path,
        baseline_lambda=0.0,
        reg_lambda=0.0001,
        num_epochs=200,
        batch_size=128,
        learning_rate=1e-4,
        early_stopping_patience=50,
        adaptive_lambda='False'  # Options: False, 'sobol', 'gradnorm'
    )
