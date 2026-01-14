import os
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from torch import float64 as fl64
from matplotlib import pyplot as plt
from SALib import ProblemSpec


def compute_full_rbf_affinity(Y, device='cpu'):
    """
    Compute fully connected RBF (Radial Basis Function) affinity matrix.

    Creates dense affinity matrix where weights are computed using RBF kernel:
    W_ij = exp(-dist(i,j)^2 / sigma^2)

    Sigma is automatically estimated from median pairwise distance.

    Args:
        Y: point coordinates [N, proj_dim]
        device: computing device ('cpu' or 'cuda')

    Returns:
        W: RBF affinity matrix [N, N] with zeros on diagonal
    """
    Y_tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    dist_matrix_all = torch.cdist(Y_tensor, Y_tensor, p=2)
    n = dist_matrix_all.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)
    all_distances = dist_matrix_all[triu_indices[0], triu_indices[1]]

    sigma_sq = torch.median(all_distances) ** 2
    W = torch.exp(- (dist_matrix_all ** 2) / sigma_sq)

    W.fill_diagonal_(0)

    print(f"  [Graph] Fully Connected RBF initialized.")
    print(f"  [Graph] Global sigma^2: {sigma_sq.item():.6f}")
    print(f"  [Graph] Matrix W size: {W.shape}")
    return W


def get_adaptive_lambda_sobol(combines_loss, nn_loss, graph_loss):
    """
    Sobol Sensitivity Analysis for adaptive lambda computation.

    Args:
        combines_loss: list of combined losses per epoch [loss_epoch0, loss_epoch1, ...]
        nn_loss: list of model losses per epoch
        graph_loss: list of graph losses per epoch

    Returns:
        list [float, float]: normalized lambda coefficients [lam_nn, lam_graph]

    """
    n_samples = 1  # can be changed to use more elements of lists
    sampling_D = 2  # as combine 2 features

    if n_samples * (sampling_D * 2 + 2) > len(combines_loss):
        print('Epochs number is too small to calculate adaptive lambda')
        return [1, 1]

    combines_loss = np.array(combines_loss)
    nn_loss = np.expand_dims(np.array(nn_loss), axis=1)
    graph_loss = np.expand_dims(np.array(graph_loss), axis=1)

    X_array = np.hstack((nn_loss, graph_loss))

    bounds = [[-100, 100] for i in range(sampling_D)]
    names = ['x{}'.format(i) for i in range(sampling_D)]

    X_array = X_array[:n_samples * (X_array.shape[1] * 2 + 2)]
    combines_loss = combines_loss[:n_samples * (X_array.shape[1] * 2 + 2)]

    sp = ProblemSpec({'names': names, 'bounds': bounds})
    sp.set_samples(X_array)
    sp.set_results(combines_loss)
    sp.analyze_sobol(calc_second_order=True)

    ST = sp.analysis['ST']
    total_disp = sum(ST)

    nn_disp = sum(ST[:nn_loss.shape[1]])
    graph_disp = sum(ST[nn_loss.shape[1]:])

    if nn_disp == 0 or graph_disp == 0:
        print(f'Lambda search failed: nn_disp={nn_disp}, graph_disp={graph_disp}')
        return [1, 1]

    lam_nn = total_disp / nn_disp
    lam_graph = total_disp / graph_disp

    if np.isnan(lam_nn) or np.isnan(lam_graph):
        print(f'Lambda search failed: nn_disp={lam_nn}, graph_disp={lam_graph}')
        return [1, 1]

    return [lam_nn / (np.nanmax([lam_nn, lam_graph])), lam_graph / (np.nanmax([lam_nn, lam_graph]))]


class GraphRegTrainer:
    """
    Class for training arbitrary machine learning models with graph regularization.

    total_loss = model_loss + lambda_graph * graph_loss
    """

    def __init__(self,
                 train_features: np.ndarray,
                 train_target: np.ndarray,
                 weights_matrix: np.ndarray,
                 base_indices: np.ndarray,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: torch.optim.Optimizer,
                 val_features: np.ndarray = None,
                 val_targets: np.ndarray = None,
                 target_metric: Callable = None,
                 num_epochs: int = 100,
                 batch_size: int = 64,
                 device: str = None,
                 cache_folder: str = None):
        """
        Args:
            train_features: training features [N, features]
            train_target: target values [N, ] or [N, output_dim]
            weights_matrix: manifold distance matrix [base_dim, base_dim]
            model: custom neural network model (nn.Module)
            criterion: loss function (e.g., nn.MSELoss, nn.CrossEntropyLoss)
            optimizer: optimizer (e.g., torch.optim.Adam)
            val_features: validation features
            val_targets:validation targets
            target_metric: evaluation metric
            num_epochs: number of epochs
            batch_size: batch size
            device: 'cuda', 'cpu' or None (auto)
            cache_folder: folder for saving models
        """
        self.best_model = None
        self.best_epoch = 0
        self.features = train_features.astype(float)
        self.target = train_target
        self.val_features = val_features
        self.val_targets = val_targets
        # fill nans for correct regularization
        self.weights_matrix = np.nan_to_num(weights_matrix)
        self.base_indices = base_indices
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        if cache_folder is not None and not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        self.convergence_history = {
            'time_spent': [],
            'model_loss': [],
            'graph_loss': [],
            'combined_loss': [],
            'val_loss': [],
            'model_lambda': [],
            'graph_lambda': []
        }

        self.device = self.init_device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.target_metric = target_metric

    def init_device(self, device: str = None):
        """
        Initialize computing device (CUDA or CPU).
        Args:
            device: device name ('cuda', 'cpu', or None for auto-detection)
        Returns:
            device: selected device name
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device

    def _compute_graph_loss_global(self, all_predictions: torch.Tensor, batch_indices=None) -> torch.Tensor:
        """
        Compute global graph regularization loss using pairwise distances.

        Computes weighted sum of prediction differences across all point pairs:
        Loss = (1/2N^2) * sum_ij (A_ij * ||f(i) - f(j)||^2)
        where A=exp(-W**2) and W is the precomputed distance matrix and f(i) are model predictions.
        Args:
            all_predictions: model predictions for all points [N, output_dim]
            batch_indices: indices of the current batch (optional). If provided,
                           computes loss only for pairs within the batch.
        Returns:
            graph_loss: scalar regularization loss value
        """
        F = all_predictions.cpu().detach().numpy()
        if batch_indices is not None:
            # loss calculates only on graph base points to save weights_matrix dimensionality
            real_indices_in_base = np.intersect1d(batch_indices,
                                                  self.base_indices)  # find batch indices which are in base
            indices_in_batch_indices = np.argwhere(np.isin(batch_indices, real_indices_in_base))[:, 0]
            F = F[indices_in_batch_indices]
            prediction_dists = pairwise_distances(F, F) ** 2

            # find valid indices to cut weights_matrix with base dimensionality
            indices_in_base = np.argwhere(np.isin(self.base_indices, real_indices_in_base))[:, 0]
            batch_weights_matrix = self.weights_matrix[indices_in_base][:, indices_in_base]
            graph_loss = np.sum(prediction_dists * np.exp(-batch_weights_matrix ** 2)) / (
                        2 * len(real_indices_in_base) ** 2)
        else:
            prediction_dists = pairwise_distances(F, F) ** 2
            graph_loss = np.sum(prediction_dists * np.exp(-self.weights_matrix ** 2)) / (2 * F.shape[0] ** 2)
        return graph_loss

    def train(self, plot_convergence: bool = False, adaptive_lambda=False,
              early_stopping_patience: int = 100, adaptive_lambda_window: int = 50):
        """
        Train the model with combined loss (model loss + graph regularization loss).

        Args:
            plot_convergence: whether to plot convergence graphs after training
            adaptive_lambda: adaptive lambda method - False (disabled) or 'sobol' (once after 10% epochs)
            early_stopping_patience: number of epochs to wait for improvement before stopping (None = no early stopping)
            adaptive_lambda_window: number of epochs to update weights of combined loss components
        Returns:
            self: trained model instance
        """
        self.model.train()

        best_val_loss = float('inf')
        increasing_counter = 0

        lam_nn = 1
        lam_graph = 1

        no_changes_epochs = 100

        if adaptive_lambda == 'sobol':
            if adaptive_lambda_window >= self.num_epochs:
                print(
                    f"Please use at least {adaptive_lambda_window} number of epochs of set adaptive_lambda as 'False'")

        start_time = time.time()
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')

            self.model.train()

            indices = np.arange(len(self.features))

            epoch_model_loss = 0.0
            epoch_graph_loss = 0.0
            epoch_combined_loss = 0.0

            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                batch_indices = indices[start_idx:min(start_idx + self.batch_size, len(indices))]
                batch_x = torch.tensor(self.features[batch_indices], dtype=fl64).to(self.device)
                batch_y = torch.tensor(self.target[batch_indices], dtype=fl64).to(self.device)
                output = self.model(batch_x)
                model_loss = self.criterion(output, batch_y.reshape_as(output))
                graph_loss = self._compute_graph_loss_global(output, batch_indices=batch_indices)

                combined_loss = lam_nn * model_loss + lam_graph * graph_loss

                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()

                epoch_model_loss += model_loss.item()
                epoch_graph_loss += graph_loss.item()
                epoch_combined_loss += combined_loss.item()
                num_batches += 1

            current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            self.convergence_history['time_spent'].append(current_time)

            avg_model_loss = epoch_model_loss / num_batches if num_batches > 0 else epoch_model_loss
            avg_graph_loss = epoch_graph_loss / num_batches if num_batches > 0 else epoch_graph_loss
            avg_combined_loss = epoch_combined_loss / num_batches if num_batches > 0 else epoch_combined_loss

            self.convergence_history['model_loss'].append(avg_model_loss)
            self.convergence_history['graph_loss'].append(avg_graph_loss)
            self.convergence_history['combined_loss'].append(avg_combined_loss)

            print(
                f'  Model loss: {avg_model_loss:.6f}, Graph loss: {avg_graph_loss:.6f}, '
                f'Combined: {avg_combined_loss:.6f}, '
                f'Lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')

            if self.val_features is not None and self.val_targets is not None:
                self.model.eval()
                with torch.no_grad():
                    val_x = torch.tensor(self.val_features, dtype=fl64).to(self.device)
                    val_y = torch.tensor(self.val_targets, dtype=fl64).to(self.device)
                    val_output = self.model(val_x)
                    val_model_loss = self.criterion(val_output, val_y.reshape_as(val_output)).item()
                self.convergence_history['val_loss'].append(val_model_loss)
                print(f'  Val   - Model loss: {val_model_loss:.6f}')

                if early_stopping_patience is not None:
                    if val_model_loss < best_val_loss:
                        best_val_loss = val_model_loss
                        self.best_model = self.model
                        self.best_epoch = epoch + 1

                    if len(self.convergence_history['val_loss']) > no_changes_epochs:
                        mean_val_loss = np.mean(self.convergence_history['val_loss'][:-no_changes_epochs])
                        if val_model_loss < mean_val_loss:
                            increasing_counter = 0
                        else:
                            increasing_counter += 1
                            print(
                                f'Patience epoch: {increasing_counter}/{early_stopping_patience}')
                            if increasing_counter >= early_stopping_patience:
                                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                                break

            if adaptive_lambda == 'sobol' and epoch % adaptive_lambda_window == 0 and epoch != 0:
                lam_nn, lam_graph = get_adaptive_lambda_sobol(self.convergence_history['combined_loss'],
                                                              self.convergence_history['model_loss'],
                                                              self.convergence_history['graph_loss'])

                self.convergence_history['model_lambda'].append(float(lam_nn))
                self.convergence_history['graph_lambda'].append(float(lam_graph))

        self.convergence_history['epoch'] = np.arange(1, len(self.convergence_history['model_loss']) + 1)
        df = pd.DataFrame({
            key: pd.Series(values)
            for key, values in self.convergence_history.items()
        })
        df.to_csv(f'{self.cache_folder}/convergence_log.csv', index=False)

        if plot_convergence:
            self._plot_convergence()

        return self

    def _plot_convergence(self):
        """
        Plot training convergence graphs.
        """
        losses = self.convergence_history['combined_loss']
        nn_losses = self.convergence_history['model_loss']
        graph_losses = self.convergence_history['graph_loss']
        val_losses = self.convergence_history['val_loss']

        fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(losses) + 1)
        axes[0].plot(epochs, losses, label='Combined Loss', color='blue', linewidth=2)
        if len(val_losses) != 0 and self.best_epoch is not None and self.best_epoch <= len(losses):
            axes[0].axvline(x=self.best_epoch, color='gray', linestyle='--',
                            linewidth=1.5, alpha=0.7, label=f'Best Val Epoch ({self.best_epoch})')
            axes[0].axhline(y=losses[self.best_epoch - 1], color='gray', linestyle='--',
                            linewidth=1, alpha=0.5)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Plot 2: Model Loss vs Graph Loss
        if len(nn_losses) != 0 and len(graph_losses) != 0:
            axes[1].plot(epochs, nn_losses, label='Model Loss', color='green', linewidth=2)
            axes[1].plot(epochs, graph_losses, label='Graph Loss', color='red', linewidth=2)

            if len(val_losses) != 0 and self.best_epoch != 0 and self.best_epoch <= len(nn_losses):
                axes[1].axvline(x=self.best_epoch, color='gray', linestyle='--',
                                linewidth=1.5, alpha=0.7, label=f'Best Val Epoch ({self.best_epoch})')
                axes[1].axhline(y=nn_losses[self.best_epoch - 1], color='green', linestyle='--',
                                linewidth=1, alpha=0.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Model Loss vs Graph Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        plt.tight_layout()

        if self.cache_folder is not None:
            save_path1 = os.path.join(self.cache_folder, "Loss_Components_Convergence.png")
            plt.savefig(save_path1, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            print(f"Convergence plot saved to {save_path1}")
        else:
            plt.show()

        if len(val_losses) > 0:
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, nn_losses, label='Training Model Loss',
                    color='blue', linewidth=2, alpha=0.8)
            val_epochs = range(1, len(val_losses) + 1)
            ax.plot(val_epochs, val_losses, label='Validation Loss',
                    color='orange', linewidth=2, alpha=0.9, linestyle='-', markersize=4)
            if self.best_epoch is not None and self.best_epoch <= len(nn_losses):
                ax.axvline(x=self.best_epoch, color='red', linestyle='--',
                           linewidth=2, alpha=0.8, label=f'Best Epoch ({self.best_epoch})')
                if self.best_epoch <= len(val_losses):
                    best_val_loss = val_losses[self.best_epoch - 1]
                else:
                    best_val_loss = val_losses[-1] if val_losses else 0
                ax.plot(self.best_epoch, best_val_loss, 'r', markersize=5,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
                        label=f'Best Val Loss: {best_val_loss:.4f}')
                ax.axhline(y=best_val_loss, color='red', linestyle=':',
                           linewidth=1, alpha=0.5)
            train_info = f'Training epochs: {len(losses)}\n'
            if self.best_epoch is not None:
                train_info += f'Best epoch: {self.best_epoch}\n'
                train_info += f'Final train loss: {nn_losses[-1]:.6f}\n'
                train_info += f'Best val loss: {best_val_loss:.6f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, train_info, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training vs Validation Convergence', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xlim(0, max(len(losses), len(val_losses)) + 1)
            plt.tight_layout()

            if self.cache_folder is not None:
                save_path2 = os.path.join(self.cache_folder, "Train_Validation_Convergence.png")
                plt.savefig(save_path2, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print(f"Train vs Validation convergence plot saved to {save_path2}")
            else:
                plt.show()
