import os
from datetime import datetime
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch import randperm, tensor
from torch.optim import Adam
from torch import float64 as fl64
from tqdm import tqdm
from matplotlib import pyplot as plt
from SALib import ProblemSpec

from Adam.Isomap import IsomapNN


def _get_adaptive_lambda(combines_loss, nn_loss, graph_loss):
    """
    :param combines_loss:  matrix m x n where m - epochs number, n - batch size with sum of nn and graph losses
    :param nn_loss: matrix m x n where m - epochs number, n - batch size with graph losses
    :param graph_loss: matrix m x n where m - epochs number, n - batch size with nn losses
    :return: list [float, float] - list with coefficients to multiply with nn loss and graph loss
    """
    n_samples = 5  # can be changed to use more elements of lists
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
                 basis_indices: np.ndarray,
                 model: nn.Module = None,
                 criterion: Callable = None,
                 optimizer: torch.optim.Optimizer = None,
                 lr: float =  1e-3,
                 target_metric: Callable = None,
                 num_epochs: int = 100,
                 batch_size: int = 64,
                 lambda_graph: float = 1.0,
                 n_neighbors: int = 5,
                 method: str = 'ensemble_knn',
                 device: str = None,
                 cache_folder: str = None,
                 verbose: bool = True,
                 precomputed_base_projections: np.ndarray = None,
                 precomputed_all_projections: np.ndarray = None):
        """
        Args:
            train_features: training features [N, features]
            train_target: target values [N, ] or [N, output_dim]
            weights_matrix: graph weight matrix [base_dim, base_dim]
            basis_indices: indices of basis points
            model: custom model (nn.Module)
            criterion: loss function
            optimizer: optimizer
            lr: learning rate
            target_metric: evaluation metric
            num_epochs: number of epochs
            batch_size: batch size
            lambda_graph: graph regularization coefficient
            n_neighbors: number of neighbors for interpolation
            method: projection method ('krr', 'ensemble_knn', 'random_forest')
            device: 'cuda', 'cpu' or None (auto)
            cache_folder: folder for saving models
            verbose: show training progress
            precomputed_base_projections: precomputed projections of basis points [base_dim, proj_dim]
                                          If provided, skips Isomap computation
            precomputed_all_projections: precomputed projections of ALL points [N, proj_dim]
                                         If provided, skips KNN RF KRR
        """
        self.features = train_features.astype(float)
        self.target = train_target

        self.weights_matrix = weights_matrix
        self.basis_indices = basis_indices
        self.source_data = train_features
        self.n_neighbors = n_neighbors
        self.method = method

        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lambda_graph = lambda_graph
        self.verbose = verbose

        self.cache_folder = cache_folder
        if cache_folder is not None and not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        self.model_name = f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_model"

        self.trained_loss_values = {
            'model_loss': None,
            'graph_loss': None,
            'combined_loss': None
        }

        self.device = self.init_device(device)

        # Use precomputed base projections if provided, otherwise compute them
        if precomputed_base_projections is not None:
            if self.verbose:
                print(f"Using precomputed base projections: {precomputed_base_projections.shape}")
            self.proj_base_features = precomputed_base_projections
        else:
            if self.verbose:
                print("Computing base projections with Isomap...")
            self.proj_base_features = self._compute_base_projections()

        # Use precomputed all projections if provided, otherwise compute them
        if precomputed_all_projections is not None:
            if self.verbose:
                print(f"Using precomputed all projections: {precomputed_all_projections.shape}")
            self.Y_all = precomputed_all_projections
        else:
            if self.verbose:
                print(f"Computing projections for all {len(train_features)} points...")
            self.Y_all = self.compute_all_projections(method=self.method)

        self.init_model(model)
        self._init_training_settings(criterion, optimizer, lr)
        self._init_target_metric(target_metric)

    def init_device(self, device: str = None):
        """
        Initialize computing device (CUDA or CPU).

        Args:
            device: device name ('cuda', 'cpu', or None for auto-detection)

        Returns
        :
            device: selected device name
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device

    def _compute_base_projections(self):
        """
        Compute projections of basis points to hidden geometry space using Isomap.

        Returns:
            projections: basis points coordinates in hidden space [base_dim, proj_dim]
        """
        proj_dim = len(self.basis_indices)

        weights_tensor = torch.tensor(self.weights_matrix,
                                      dtype=torch.float64).to(self.device)

        isomap = IsomapNN(
            weights_initial_assumption=weights_tensor,
            n_components=proj_dim,
            n_neighbors=self.n_neighbors
        )

        projections = isomap.fit_transform(weights_tensor)

        return projections.detach().cpu().numpy()

    def init_model(self, model):
        """
        Initialize neural network model.

        Args:
            model: custom nn.Module or None for default architecture
        """
        if model is not None:
            self.model = model
        else:
            input_dim = self.features.shape[1]
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512, dtype=fl64),
                nn.ReLU(),
                nn.Linear(512, 256, dtype=fl64),
                nn.ReLU(),
                nn.Linear(256, 64, dtype=fl64),
                nn.ReLU(),
                nn.Linear(64, 1, dtype=fl64)
            )

        self.model = self.model.to(self.device)

    def _init_training_settings(self, criterion, optimizer, lr):
        """
        Initialize loss criterion and optimizer.

        Args:
            criterion: loss function or None for MSELoss
            optimizer: optimizer or None for Adam
            lr: learning rate
        """
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=1e-3, eps=1e-4)
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion


    def _init_target_metric(self, target_metric):
        """
        Initialize evaluation metric.

        Args:
            target_metric: metric function or None
        """
        self.target_metric = target_metric

    def _compute_graph_loss(self, predictions: np.ndarray, batch_indices: np.ndarray) -> float:
        """
        Compute graph regularization loss for a batch of predictions.

        Args:
            predictions: model predictions for batch [batch_size, output_dim]
            batch_indices: indices of batch elements in dataset

        Returns:
            loss: graph loss value
        """
        from sklearn.metrics.pairwise import euclidean_distances

        Y_batch = self.Y_all[batch_indices]
        distances = euclidean_distances(Y_batch, Y_batch)
        sigma = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
        W_batch = distances / np.max(distances) if np.max(distances) > 0 else distances
        #W_batch = np.exp(-distances ** 2 / (2 * sigma ** 2))
        D_batch = np.diag(np.sum(W_batch, axis=1))
        L_batch = D_batch - W_batch

        part_1 = np.dot(predictions.T, L_batch)
        loss = np.dot(part_1, predictions)
        return loss.reshape(-1)[0]

    def train(self, plot_convergence: bool = False, adaptive_lambda: bool = False):
        """
        Train the model with combined loss (model loss + graph regularization loss).

        Args:
            plot_convergence: whether to plot convergence graphs after training
            adaptive_lambda: whether to use adaptive lambda (computed once after 10% of epochs)

        Returns:
            self: trained model instance
        """
        self.model.train()
        model_losses = []
        graph_losses = []
        combined_losses = []

        lam_nn = 1
        lam_graph = self.lambda_graph
        lmds_epochs = int(self.num_epochs * 0.1)

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.num_epochs}')

            indices = randperm(len(self.features)).numpy()

            epoch_model_losses = []
            epoch_graph_losses = []
            epoch_combined_losses = []

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                batch_x = torch.tensor(self.features[batch_indices], dtype=fl64).to(self.device)


                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    batch_y = torch.tensor(self.target[batch_indices], dtype=torch.long).to(self.device)
                else:
                    batch_y = torch.tensor(self.target[batch_indices], dtype=fl64).to(self.device)

                self.optimizer.zero_grad()

                output = self.model(batch_x)

                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    model_loss = self.criterion(output, batch_y)
                else:
                    model_loss = self.criterion(output, batch_y.reshape_as(output))

                predictions_np = output.detach().cpu().numpy()
                graph_loss_value = self._compute_graph_loss(predictions_np, batch_indices)
                graph_loss = torch.tensor(graph_loss_value, dtype=fl64).to(self.device)

                combined_loss = lam_nn * model_loss + lam_graph * graph_loss

                combined_loss.backward()
                self.optimizer.step()

                epoch_model_losses.append(model_loss.item())
                epoch_graph_losses.append(graph_loss_value)
                epoch_combined_losses.append(combined_loss.item())

            model_losses.append(np.mean(epoch_model_losses))
            graph_losses.append(np.mean(epoch_graph_losses))
            combined_losses.append(np.mean(epoch_combined_losses))

            if adaptive_lambda and epoch == lmds_epochs:
                lam_nn, lam_graph = _get_adaptive_lambda(combined_losses, model_losses, graph_losses)



            if self.verbose:
                print(f'  Model loss: {model_losses[-1]:.6f}, Graph loss: {graph_losses[-1]:.6f}, Combined: {combined_losses[-1]:.6f}')
                print(f' Adaptive lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')
        self.trained_loss_values['model_loss'] = model_losses
        self.trained_loss_values['graph_loss'] = graph_losses
        self.trained_loss_values['combined_loss'] = combined_losses

        if plot_convergence:
            self._plot_convergence(combined_losses, None, model_losses, graph_losses)

        return self

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            test_features: test features [N_test, features]

        Returns:
            predictions: model predictions [N_test, output_dim]
        """
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(test_features, dtype=fl64).to(self.device)
            predictions = self.model(test_tensor)
        return predictions.cpu().numpy()

    def evaluate(self, test_features: np.ndarray, test_target: np.ndarray) -> float:
        """
        Evaluate model quality on test data.

        Args:
            test_features: test features [N_test, features]
            test_target: test target values [N_test, ] or [N_test, output_dim]

        Returns:
            metric_value: evaluation metric value
        """
        predictions = self.predict(test_features)

        if self.target_metric is not None:
            metric_value = self.target_metric(test_target, predictions)
        else:
            from sklearn.metrics import mean_squared_error
            metric_value = mean_squared_error(test_target, predictions)

        return metric_value

    def save_weights(self, path: str = None):
        """
        Save model weights to file.

        Args:
            path: path to save weights or None for default location
        """
        if path is None:
            if self.cache_folder is not None:
                path = os.path.join(self.cache_folder, f"{self.model_name}.pth")
            else:
                path = f"{self.model_name}.pth"

        torch.save(self.model.state_dict(), path)
        if self.verbose:
            print(f"Model weights saved to {path}")

    def _plot_convergence(self, losses, lmds_epoch, nn_losses=None, graph_losses=None):
        """
        Plot training convergence graphs.

        Args:
            losses: combined loss values per epoch
            lmds_epoch: lambda values per epoch (not used in current implementation)
            nn_losses: model loss values per epoch
            graph_losses: graph loss values per epoch
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(losses) + 1)

        axes[0].plot(epochs, losses, label='Combined Loss', color='blue')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Loss')
        axes[0].legend()
        axes[0].grid(True)

        if nn_losses is not None and graph_losses is not None:
            axes[1].plot(epochs, nn_losses, label='Model Loss', color='green')
            axes[1].plot(epochs, graph_losses, label='Graph Loss', color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Model Loss vs Graph Loss')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()

        if self.cache_folder is not None:
            save_path = os.path.join(self.cache_folder, f"{self.model_name}_convergence.png")
            plt.savefig(save_path)
            if self.verbose:
                print(f"Convergence plot saved to {save_path}")

        plt.show()