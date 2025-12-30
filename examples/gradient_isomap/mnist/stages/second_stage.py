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
from utils.utils import check_required_files, restore_data_from_metadata, set_global_seed, \
                               load_data_from_folder
from utils.Projector import Projector

#Set the name of your run folder from Stage 1
RUN_FOLDER_NAME = 'run_20251210_150023_n2000'

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


def train_baseline_model(X_train, y_train, X_val, y_val, X_test, y_test,
                        model_name, cache_folder, num_epochs, batch_size, learning_rate, early_stopping_patience):
    """
    Train baseline model with simple PyTorch loop

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        model_name: name for logging
        cache_folder: folder to save results
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate
        early_stopping_patience: patience for early stopping

    Returns:
        dict with metrics and predictions
    """
    print(f"\n--- Training {model_name} (Baseline) ---")
    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Batch size: {batch_size}")

    # Create model, criterion, optimizer
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    val_loss_window = []

    for epoch in range(num_epochs):
        model.train()

        indices = torch.randperm(len(X_train))
        num_batches = (len(indices) + batch_size - 1) // batch_size

        all_predictions = []
        all_targets = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]

            batch_x = torch.tensor(X_train[batch_indices], dtype=torch.float64).to(device)
            batch_y = torch.tensor(y_train[batch_indices], dtype=torch.long).to(device)

            output = model(batch_x)

            all_predictions.append(output)
            all_targets.append(batch_y)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        train_loss = criterion(all_predictions, all_targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Track loss
        train_losses.append(train_loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float64).to(device)
            val_y = torch.tensor(y_val, dtype=torch.long).to(device)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
            val_losses.append(val_loss)

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                train_x = torch.tensor(X_train, dtype=torch.float64).to(device)
                train_y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
                train_output = model(train_x)
                train_pred = torch.argmax(train_output, dim=1).cpu().numpy()
                train_acc = accuracy_score(y_train, train_pred)
                train_accuracies.append(train_acc)

                val_pred = torch.argmax(val_output, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_pred)
                val_accuracies.append(val_acc)

                if epoch % 10 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}')
                    print(f'  Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss:.6f}')
                    print(f'  Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1

        val_loss_window.append(val_loss)

        if len(val_loss_window) > 1000:
            val_loss_window.pop(0)

        if len(val_loss_window) >= min(1000, epoch + 1):
            median_val_loss = np.median(val_loss_window)

            if val_loss < median_val_loss:
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f'\nEarly stopping at epoch {epoch + 1}')
                    print(f'Val loss above median for {early_stopping_patience} epochs')
                    print(f'Best model was at epoch {best_epoch} with val loss {best_val_loss:.6f}')
                    break

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model weights from epoch {best_epoch}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Validation
        val_x = torch.tensor(X_val, dtype=torch.float64).to(device)
        val_output = model(val_x)
        val_pred = torch.argmax(val_output, dim=1).cpu().numpy()
        val_accuracy = accuracy_score(y_val, val_pred)

        # Test
        test_x = torch.tensor(X_test, dtype=torch.float64).to(device)
        test_output = model(test_x)
        test_pred = torch.argmax(test_output, dim=1).cpu().numpy()
        test_accuracy = accuracy_score(y_test, test_pred)

        print(f"\n{model_name} Final Results:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, test_pred, digits=4))

    # Save model
    if cache_folder is not None:
        os.makedirs(cache_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cache_folder, 'best_model.pth'))

    return {
        'accuracy': test_accuracy,
        'val_accuracy': val_accuracy,
        'predictions': test_pred,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch
    }


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
        base_indices=fps_indices,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lambda_graph=lambda_graph,
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
    if len(baseline_train_losses) > 0:
        ax1.plot(baseline_train_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
    if len(reg_train_losses) > 0:
        ax1.plot(reg_train_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    if len(baseline_train_losses) > 0 or len(reg_train_losses) > 0:
        ax1.set_yscale('log')

    # Top-right: Validation Loss
    ax2 = axes[0, 1]
    if len(baseline_val_losses) > 0:
        ax2.plot(baseline_val_losses, label='Baseline', linewidth=2, alpha=0.8, color='blue')
    if len(reg_val_losses) > 0:
        ax2.plot(reg_val_losses, label='Regularized', linewidth=2, alpha=0.8, color='red')

    if len(baseline_val_losses) > 0:
        baseline_best_epoch = baseline_results.get('best_epoch', None)
        if baseline_best_epoch is not None and baseline_best_epoch <= len(baseline_val_losses):
            # Vertical line
            ax2.axvline(x=baseline_best_epoch-1, color='blue', linestyle='--',
                       linewidth=1, alpha=0.6, label=f'Baseline Best (epoch {baseline_best_epoch})')
            # Horizontal line
            best_val_loss = baseline_val_losses[baseline_best_epoch-1]
            ax2.axhline(y=best_val_loss, color='blue', linestyle='--',
                       linewidth=1, alpha=0.6)

    if len(reg_val_losses) > 0:
        reg_best_epoch = reg_results.get('best_epoch', None)
        if reg_best_epoch is not None and reg_best_epoch <= len(reg_val_losses):
            ax2.axvline(x=reg_best_epoch-1, color='red', linestyle='--',
                       linewidth=1, alpha=0.6, label=f'Regularized Best (epoch {reg_best_epoch})')
            best_val_loss = reg_val_losses[reg_best_epoch-1]
            ax2.axhline(y=best_val_loss, color='red', linestyle='--',
                       linewidth=1, alpha=0.6)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    if len(baseline_val_losses) > 0 or len(reg_val_losses) > 0:
        ax2.set_yscale('log')

    # Bottom-left: Training Accuracy
    ax3 = axes[1, 0]
    if len(baseline_train_accs) > 0 and len(baseline_train_losses) > 0:
        baseline_acc_interval = max(1, len(baseline_train_losses) // len(baseline_train_accs))
        baseline_epochs_acc = [i * baseline_acc_interval for i in range(len(baseline_train_accs))]
        ax3.plot(baseline_epochs_acc, baseline_train_accs, label='Baseline',
                 linewidth=2, alpha=0.8, color='blue', marker='o', markersize=3)

    if len(reg_train_accs) > 0 and len(reg_train_losses) > 0:
        reg_acc_interval = max(1, len(reg_train_losses) // len(reg_train_accs))
        reg_epochs_acc = [i * reg_acc_interval for i in range(len(reg_train_accs))]
        ax3.plot(reg_epochs_acc, reg_train_accs, label='Regularized',
                 linewidth=2, alpha=0.8, color='red', marker='s', markersize=3)

    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Training Accuracy', fontsize=11)
    ax3.set_title('Training Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    ax4 = axes[1, 1]
    if len(baseline_val_accs) > 0 and len(baseline_val_losses) > 0:
        baseline_acc_interval = max(1, len(baseline_val_losses) // len(baseline_val_accs))
        baseline_epochs_acc = [i * baseline_acc_interval for i in range(len(baseline_val_accs))]
        ax4.plot(baseline_epochs_acc, baseline_val_accs, label='Baseline',
                 linewidth=2, alpha=0.8, color='blue', marker='o', markersize=3)

    if len(reg_val_accs) > 0 and len(reg_val_losses) > 0:
        reg_acc_interval = max(1, len(reg_val_losses) // len(reg_val_accs))
        reg_epochs_acc = [i * reg_acc_interval for i in range(len(reg_val_accs))]
        ax4.plot(reg_epochs_acc, reg_val_accs, label='Regularized',
                 linewidth=2, alpha=0.8, color='red', marker='s', markersize=3)

    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Validation Accuracy', fontsize=11)
    ax4.set_title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    ax4.text(0.02, 0.98,
             f'Final Val Acc:\nBaseline: {baseline_val_acc:.4f}\nRegularized: {reg_val_acc:.4f}',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    improvement = (reg_test_acc - baseline_test_acc) * 100
    fig.suptitle(f'MNIST Classification: Baseline vs Regularized Comparison\n' +
                 f'Test Accuracy: Baseline: {baseline_test_acc:.4f}, Regularized: {reg_test_acc:.4f} '
                 f'(Improvement: {improvement:+.2f}%)',
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
                               reg_lambda: float = 0.00001,
                               num_epochs: int = 200,
                               batch_size: int = 128,
                               learning_rate: float = 1e-6,
                               early_stopping_patience: int = 100,
                               adaptive_lambda: Union[bool, str] = False):
    """
    Main function to train MNIST classifier with graph regularization

    Args:
        folder_path: path to folder with data from Stage 1
        reg_lambda: lambda for regularized model
        num_epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        early_stopping_patience: patience for early stopping
        adaptive_lambda: 'sobol', False
    """
    required_files = [
        'fps_indices.npy',
        'best_distance_matrix.npy',
        'train_projections.npy',
        'base_projections.npy',
        'experiment_metadata.json'
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
    metadata_path = os.path.join(folder_path, 'experiment_metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            saved_seed = metadata.get('random_seed')
            print(f"\nFound saved seed: {saved_seed}")
    except FileNotFoundError:
        print("\nMetadata file not found.")

    set_global_seed(saved_seed)
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


    n_base = len(fps_indices)
    weights_matrix = Projector.reconstruct_distance_matrix(best_distances_matrix, n_base)

    print("\n" + "="*60)
    print("TRAINING CLASSIFIER WITH GRAPH REGULARIZATION")
    print("="*60)
    print(f"Using precomputed projections")
    set_global_seed(saved_seed)
    baseline_results = train_baseline_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name="Baseline",
        cache_folder=os.path.join(experiment_folder, 'baseline'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )
    set_global_seed(saved_seed)
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

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    baseline_test_acc = baseline_results['accuracy']
    baseline_val_acc = baseline_results['val_accuracy']
    reg_test_acc = reg_results['accuracy']
    reg_val_acc = reg_results['val_accuracy']

    print(f"Baseline Accuracy:    {baseline_test_acc:.4f}")
    print(f"Regularized Accuracy: {reg_test_acc:.4f}")

    print("\n  Creating comparison visualization...")
    viz_path = os.path.join(experiment_folder, 'mnist_comparison.png')
    create_mnist_comparison_visualization(
        baseline_results, reg_results, viz_path
    )

    baseline_params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'early_stopping_patience': early_stopping_patience,
        'adaptive_lambda': {'method': 'disabled'},
        'accuracy_check_interval': 10,
        'best_epoch': baseline_results.get('best_epoch', num_epochs)
    }

    # Determine adaptive lambda config
    if adaptive_lambda == 'sobol':
        adaptive_lambda_config = {'method': 'sobol'}
    else:
        adaptive_lambda_config = {'method': 'disabled'}

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

    lambda_history = reg_results['trainer'].trained_loss_values.get('lambda_history', None)
    if lambda_history is not None:
        reg_params['lambda_history'] = lambda_history

    results = {
        'baseline_test_accuracy': float(baseline_test_acc),
        'baseline_val_accuracy': float(baseline_val_acc),
        'regularized_test_accuracy': float(reg_test_acc),
        'regularized_val_accuracy': float(reg_val_acc),
        'accuracy_improvement_percent': float((reg_test_acc - baseline_test_acc) * 100),
        'n_base_points': int(n_base),
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
    experiment_dir = os.path.abspath(os.path.join(script_dir, '..'))
    folder_path = os.path.join(experiment_dir, 'outputs', RUN_FOLDER_NAME)

    print(f"Experiment dir: {experiment_dir}")
    print(f"Looking for data in: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"\nError: Folder not found: {folder_path}")
        print("\nPlease update 'RUN_FOLDER_NAME' in this script or run first_stage.py")
        sys.exit(1)

    results = mnist_graph_regularization(
        folder_path=folder_path,
        reg_lambda=1,
        num_epochs=200,
        batch_size=128,
        learning_rate=1e-4,
        early_stopping_patience=150,
        adaptive_lambda='sobol'  # Options: False, 'sobol'
    )
