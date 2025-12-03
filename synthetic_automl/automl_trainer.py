
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import randperm, tensor
from torch import float64 as fl64
from torch.optim import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .lambda_optimizers import BaseLambdaOptimizer, create_optimizer


# ============== Вспомогательные функции из оригинального кода ==============

def project_ensemble_knn(X_sparse, Y_sparse, X_all):
    """Ensemble KNN regression для проекции точек"""
    estimators = [
        ('knn5', KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)),
        ('knn10', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)),
        ('knn15', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)),
    ]
    
    n_components = Y_sparse.shape[1]
    Y_projected = np.zeros((len(X_all), n_components))
    
    for dim in range(n_components):
        ensemble = VotingRegressor(estimators, n_jobs=-1)
        ensemble.fit(X_sparse, Y_sparse[:, dim])
        Y_projected[:, dim] = ensemble.predict(X_all)
    
    return Y_projected


class RegressionModel(nn.Module):
    """Простая нейронная сеть для регрессии (как в second_stage.py)"""
    
    def __init__(self, input_dim: int = 3, hidden_dims: list = [64, 32], output_dim: int = 1):
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


class AutoMLGraphRegTrainer:
    """
    Trainer с поддержкой различных методов оптимизации lambda.
    
    Основан на GraphRegTrainer, но с подключаемым lambda-оптимизатором.
    """
    
    def __init__(self,
                 train_features: np.ndarray,
                 train_target: np.ndarray,
                 weights_matrix: np.ndarray,
                 basis_indices: np.ndarray,
                 lambda_optimizer: BaseLambdaOptimizer,
                 model: nn.Module = None,
                 criterion: Callable = None,
                 optimizer: torch.optim.Optimizer = None,
                 lr: float = 1e-3,
                 num_epochs: int = 100,
                 batch_size: int = 64,
                 n_neighbors: int = 5,
                 device: str = None,
                 verbose: bool = True,
                 precomputed_base_projections: np.ndarray = None,
                 precomputed_all_projections: np.ndarray = None):
        """
        Args:
            train_features: training features [N, features]
            train_target: target values [N, ] or [N, output_dim]
            weights_matrix: graph weight matrix [base_dim, base_dim]
            basis_indices: indices of basis points
            lambda_optimizer: объект BaseLambdaOptimizer для управления lambda
            model: custom model (nn.Module)
            criterion: loss function
            optimizer: optimizer
            lr: learning rate
            num_epochs: number of epochs
            batch_size: batch size
            n_neighbors: number of neighbors for interpolation
            device: 'cuda', 'cpu' or None (auto)
            verbose: show training progress
            precomputed_base_projections: precomputed projections of basis points
            precomputed_all_projections: precomputed projections of ALL points
        """
        self.features = train_features.astype(float)
        self.target = train_target
        self.weights_matrix = weights_matrix
        self.basis_indices = basis_indices
        self.n_neighbors = n_neighbors
        
        self.lambda_optimizer = lambda_optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        
        # Device
        self.device = self._init_device(device)
        
        # Projections
        if precomputed_base_projections is not None:
            self.proj_base_features = precomputed_base_projections
        else:
            raise ValueError("precomputed_base_projections is required")
        
        if precomputed_all_projections is not None:
            self.Y_all = precomputed_all_projections
        else:
            raise ValueError("precomputed_all_projections is required")
        
        # Model
        self._init_model(model)
        self._init_training_settings(criterion, optimizer, lr)
        
        # Training history
        self.trained_loss_values = {
            'model_loss': [],
            'graph_loss': [],
            'combined_loss': [],
            'val_loss': [],
            'lam_nn': [],
            'lam_graph': []
        }
        
        # Best model tracking
        self.best_model_state = None
        self.best_epoch = 0
        self.best_val_loss = float('inf')
    
    def _init_device(self, device: str = None) -> str:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _init_model(self, model: nn.Module = None) -> None:
        if model is not None:
            self.model = model
        else:
            input_dim = self.features.shape[1]
            self.model = RegressionModel(input_dim=input_dim)
        self.model = self.model.to(self.device)
    
    def _init_training_settings(self, criterion, optimizer, lr) -> None:
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=lr, eps=1e-4)
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
    
    def _compute_graph_loss(self, predictions: torch.Tensor,
                           batch_indices: np.ndarray) -> torch.Tensor:
        """
        Compute graph regularization with symmetric normalized Laplacian.
        L_sym = D^(-1/2) (D - W) D^(-1/2) = I - D^(-1/2) W D^(-1/2)
        """
        Y_batch = torch.tensor(self.Y_all[batch_indices], dtype=torch.float64).to(self.device)
        
        distances_sq = torch.cdist(Y_batch, Y_batch, p=2) ** 2
        nonzero_dists = distances_sq[distances_sq > 0]
        if len(nonzero_dists) > 0:
            sigma = torch.median(nonzero_dists).sqrt()
        else:
            sigma = torch.tensor(1.0, device=self.device)
        
        W_batch = torch.exp(-distances_sq / (2 * sigma ** 2))
        W_batch = W_batch - torch.diag(torch.diag(W_batch))
        
        D_diag = torch.sum(W_batch, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D_diag + 1e-10, -0.5))
        
        I = torch.eye(W_batch.shape[0], device=self.device, dtype=torch.float64)
        L_sym = I - D_inv_sqrt @ W_batch @ D_inv_sqrt
        
        loss = torch.trace(predictions.T @ L_sym @ predictions)
        n = predictions.shape[0]
        loss = loss / n
        return loss
    
    def train(self,
              val_features: np.ndarray = None,
              val_target: np.ndarray = None,
              early_stopping_patience: int = None) -> 'AutoMLGraphRegTrainer':
        """
        Train the model with lambda-optimizer controlled regularization.
        
        Args:
            val_features: validation features for early stopping
            val_target: validation target for early stopping
            early_stopping_patience: number of epochs to wait for improvement
        
        Returns:
            self: trained model instance
        """
        self.model.train()
        
        # Reset optimizer
        self.lambda_optimizer.reset()
        
        # Early stopping setup
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Get current lambdas from optimizer
            lam_nn, lam_graph = self.lambda_optimizer.get_lambdas()
            
            if self.verbose and epoch % max(1, self.num_epochs // 20) == 0:
                print(f'Epoch {epoch + 1}/{self.num_epochs} | '
                      f'lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')
            
            indices = randperm(len(self.features)).numpy()
            
            epoch_model_losses = []
            epoch_graph_losses = []
            epoch_combined_losses = []
            
            # Training phase
            self.model.train()
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_x = torch.tensor(self.features[batch_indices], dtype=fl64).to(self.device)
                batch_y = torch.tensor(self.target[batch_indices], dtype=fl64).to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(batch_x)
                
                model_loss = self.criterion(output, batch_y.reshape_as(output))
                graph_loss = self._compute_graph_loss(output, batch_indices)
                
                combined_loss = lam_nn * model_loss + lam_graph * graph_loss
                
                combined_loss.backward()
                self.optimizer.step()
                
                epoch_model_losses.append(model_loss.item())
                epoch_graph_losses.append(graph_loss.item())
                epoch_combined_losses.append(combined_loss.item())
            
            # Record epoch losses
            mean_model_loss = np.mean(epoch_model_losses)
            mean_graph_loss = np.mean(epoch_graph_losses)
            mean_combined_loss = np.mean(epoch_combined_losses)
            
            self.trained_loss_values['model_loss'].append(mean_model_loss)
            self.trained_loss_values['graph_loss'].append(mean_graph_loss)
            self.trained_loss_values['combined_loss'].append(mean_combined_loss)
            self.trained_loss_values['lam_nn'].append(lam_nn)
            self.trained_loss_values['lam_graph'].append(lam_graph)
            
            # Update lambda optimizer
            self.lambda_optimizer.update(
                epoch=epoch,
                model_loss=mean_model_loss,
                graph_loss=mean_graph_loss,
                model=self.model,
                num_epochs=self.num_epochs
            )
            
            # Validation phase
            if val_features is not None and val_target is not None:
                val_loss = self._compute_validation_loss(val_features, val_target)
                self.trained_loss_values['val_loss'].append(val_loss)
                
                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        patience_counter = 0
                        self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        self.best_epoch = epoch + 1
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if self.verbose:
                                print(f'\nEarly stopping at epoch {epoch + 1}')
                            break
        
        return self
    
    def _compute_validation_loss(self, val_features: np.ndarray, val_target: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            val_x = torch.tensor(val_features, dtype=fl64).to(self.device)
            val_y = torch.tensor(val_target, dtype=fl64).to(self.device)
            output = self.model(val_x)
            val_loss = self.criterion(output, val_y.reshape_as(output))
        return val_loss.item()
    
    def load_best_weights(self) -> None:
        """Load the best model weights saved during training"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Loaded best model from epoch {self.best_epoch}")
    
    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_features, dtype=fl64).to(self.device)
            predictions = self.model(test_tensor)
        return predictions.cpu().numpy()
    
    def evaluate(self, test_features: np.ndarray, test_target: np.ndarray) -> Dict[str, float]:
        """Evaluate model quality on test data"""
        predictions = self.predict(test_features).flatten()
        test_target_flat = test_target.flatten()
        
        return {
            'mse': mean_squared_error(test_target_flat, predictions),
            'mae': mean_absolute_error(test_target_flat, predictions),
            'r2': r2_score(test_target_flat, predictions)
        }


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights_matrix: np.ndarray,
    fps_indices: np.ndarray,
    base_projections: np.ndarray,
    train_projections: np.ndarray,
    optimizer_name: str,
    optimizer_kwargs: dict = None,
    num_epochs: int = 15000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = None,
    save_folder: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single experiment with specified lambda optimizer.
    
    Returns:
        Dictionary with metrics, training history, and timing
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    lambda_optimizer = create_optimizer(optimizer_name, **optimizer_kwargs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: {lambda_optimizer.name}")
        print(f"{'='*60}")

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    model = RegressionModel(input_dim=X_train.shape[1], hidden_dims=[64, 32], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = AutoMLGraphRegTrainer(
        train_features=X_train,
        train_target=y_train,
        weights_matrix=weights_matrix,
        basis_indices=fps_indices,
        lambda_optimizer=lambda_optimizer,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
        precomputed_base_projections=base_projections,
        precomputed_all_projections=train_projections
    )

    start_time = time.time()
    trainer.train(
        val_features=X_val,
        val_target=y_val,
        early_stopping_patience=early_stopping_patience
    )
    training_time = time.time() - start_time

    trainer.load_best_weights()

    val_metrics = trainer.evaluate(X_val, y_val)
    test_metrics = trainer.evaluate(X_test, y_test)

    results = {
        'optimizer_name': lambda_optimizer.name,
        'val_mse': val_metrics['mse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'test_mse': test_metrics['mse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'training_time_seconds': training_time,
        'best_epoch': trainer.best_epoch,
        'num_epochs_run': len(trainer.trained_loss_values['model_loss']),
        'history': trainer.trained_loss_values,
        'lambda_history': lambda_optimizer.get_history()
    }
    
    if verbose:
        print(f"\nResults for {lambda_optimizer.name}:")
        print(f"  Val MSE:  {val_metrics['mse']:.6f}")
        print(f"  Test MSE: {test_metrics['mse']:.6f}")
        print(f"  Test R²:  {test_metrics['r2']:.6f}")
        print(f"  Time:     {training_time:.1f}s")
        print(f"  Best epoch: {trainer.best_epoch}")
    
    # Save results if folder specified
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        
        # Save metrics
        metrics_to_save = {k: v for k, v in results.items() 
                         if k not in ['history', 'lambda_history']}
        with open(os.path.join(save_folder, 'metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save history
        np.save(os.path.join(save_folder, 'history.npy'), results['history'])
        np.save(os.path.join(save_folder, 'lambda_history.npy'), results['lambda_history'])
        
        # Plot convergence
        _plot_convergence(results, save_folder)
    
    return results


def _plot_convergence(results: Dict[str, Any], save_folder: str) -> None:
    """Plot and save convergence curves"""
    history = results['history']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['model_loss']) + 1)

    ax1 = axes[0]
    ax1.plot(epochs, history['model_loss'], label='Model Loss', alpha=0.8)
    ax1.plot(epochs, history['graph_loss'], label='Graph Loss', alpha=0.8)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', alpha=0.8, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"{results['optimizer_name']} - Losses")
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs, history['lam_nn'], label='λ_nn', alpha=0.8)
    ax2.plot(epochs, history['lam_graph'], label='λ_graph', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Lambda')
    ax2.set_title(f"{results['optimizer_name']} - Lambda Values")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined loss
    ax3 = axes[2]
    ax3.plot(epochs, history['combined_loss'], label='Combined Loss', color='purple', alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title(f"{results['optimizer_name']} - Combined Loss")
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
