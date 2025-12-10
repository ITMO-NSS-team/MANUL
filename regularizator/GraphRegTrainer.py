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
from utils.Projector import project_krr_optimized, project_ensemble_knn, project_random_forest


def compute_global_sigma(Y_all, k=10, device="cpu"):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(Y_all)
    distances, _ = nbrs.kneighbors(Y_all)

    nonzero = distances[:, 1:].reshape(-1)
    sigma_sq = np.median(nonzero) ** 2

    return torch.tensor(sigma_sq, dtype=torch.float64, device=device)


def _get_adaptive_lambda_sobol(combines_loss, nn_loss, graph_loss):
    """
    Sobol Sensitivity Analysis for adaptive lambda computation.

    :param combines_loss:  matrix m x n where m - epochs number, n - batch size with sum of nn and graph losses
    :param nn_loss: matrix m x n where m - epochs number, n - batch size with graph losses
    :param graph_loss: matrix m x n where m - epochs number, n - batch size with nn losses
    :return: list [float, float] - list with coefficients to multiply with nn loss and graph loss
    """
    sampling_D = 2  # as combine 2 features

    available_epochs = len(combines_loss)
    max_n_samples = available_epochs // (sampling_D * 2 + 2)

    # Use at least 5 samples if possible, otherwise use maximum available
    desired_n_samples = 32
    n_samples = min(desired_n_samples, max_n_samples)

    min_required_samples = 10
    if n_samples < min_required_samples:
        required_epochs = min_required_samples * (sampling_D * 2 + 2)
        print(f'  [Sobol] Not enough epochs for adaptive lambda computation')
        print(f'  [Sobol] Available: {available_epochs} epochs, Required: {required_epochs} epochs (for {min_required_samples} samples)')
        print(f'  [Sobol] Skipping adaptation, using default lambdas [1.0, 1.0]')
        return [1, 1]

    if n_samples < desired_n_samples:
        print(f'  [Sobol] Using {n_samples} samples (reduced from {desired_n_samples} due to limited epochs: {available_epochs})')

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
        print(f'  [Sobol] Lambda search failed: nn_disp={nn_disp}, graph_disp={graph_disp}')
        return [1, 1e-6]

    lam_nn = total_disp / nn_disp
    lam_graph = total_disp / graph_disp

    if np.isnan(lam_nn) or np.isnan(lam_graph):
        print(f'  [Sobol] Lambda search failed: nn_disp={lam_nn}, graph_disp={lam_graph}')
        return [1, 1e-6]

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
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: torch.optim.Optimizer,
                 task_type: str = None,
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
            model: custom neural network model (nn.Module)
            criterion: loss function (e.g., nn.MSELoss, nn.CrossEntropyLoss)
            optimizer: optimizer (e.g., torch.optim.Adam)
            task_type: optional explicit task type ('regression' or 'classification').
                      If None, will be inferred from criterion
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
                                          If provided, skips expensive Isomap computation
            precomputed_all_projections: precomputed projections of ALL points [N, proj_dim]
                                         If provided, skips expensive KNN interpolation
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
            'combined_loss': None,
            'lambda_history': None
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

        self.model = model.to(self.device)
        self._init_training_settings(criterion, optimizer)
        self._init_target_metric(target_metric)
        self._init_task_type(criterion, task_type)
        self.sigma_sq_global = compute_global_sigma(self.Y_all)

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


    def compute_all_projections(self, method='ensemble_knn'):
        """
        Projects ALL points from Euclidean space to hidden geometry.

        Args:
            method: projection method ('krr', 'ensemble_knn', 'random_forest')

        Returns:
            proj_all: projections of all points [N, proj_dim]
        """
        X_basis = self.source_data[self.basis_indices]
        Y_basis = self.proj_base_features
        X_all = self.source_data

        if method == 'krr':
            Y_all = project_krr_optimized(X_basis, Y_basis, X_all, batch_size=self.batch_size)
        elif method == 'ensemble_knn':
            Y_all = project_ensemble_knn(X_basis, Y_basis, X_all)
        elif method == 'random_forest':
            Y_all = project_random_forest(X_basis, Y_basis, X_all, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        Y_all[self.basis_indices] = Y_basis

        return Y_all

    def _infer_task_type(self, criterion) -> str:
        """
        Infer task type from criterion.

        Args:
            criterion: loss function

        Returns:
            'classification', 'regression', or 'custom'
        """
        classification_losses = (
            nn.CrossEntropyLoss,
            nn.NLLLoss,
            nn.BCELoss,
            nn.BCEWithLogitsLoss,
            nn.MultiLabelSoftMarginLoss,
            nn.MultiMarginLoss,
        )

        regression_losses = (
            nn.MSELoss,
            nn.L1Loss,
            nn.SmoothL1Loss,
            nn.HuberLoss,
        )

        if isinstance(criterion, classification_losses):
            return 'classification'
        elif isinstance(criterion, regression_losses):
            return 'regression'
        else:
            return 'custom'

    def _init_task_type(self, criterion, task_type):
        """
        Initialize task type (classification or regression).

        Args:
            criterion: loss function
            task_type: explicit task type or None for auto-inference
        """
        inferred_type = self._infer_task_type(criterion)

        if task_type is None:
            if inferred_type == 'custom':
                raise ValueError(
                    "Custom loss function detected. Please explicitly specify "
                    "task_type='regression' or task_type='classification'"
                )
            self.task_type = inferred_type
        else:
            if task_type not in ['regression', 'classification']:
                raise ValueError(
                    f"task_type must be 'regression' or 'classification', got {task_type}"
                )
            self.task_type = task_type

            # Warn if explicit task_type contradicts inferred type
            if inferred_type != 'custom' and inferred_type != task_type:
                import warnings
                warnings.warn(
                    f"Explicit task_type='{task_type}' contradicts inferred type "
                    f"'{inferred_type}' from criterion. Using explicit task_type."
                )

    def _init_training_settings(self, criterion, optimizer):
        """
        Initialize loss criterion and optimizer.

        Args:
            criterion: loss function
            optimizer: optimizer
        """
        self.criterion = criterion
        self.optimizer = optimizer


    def _init_target_metric(self, target_metric):
        """
        Initialize evaluation metric.

        Args:
            target_metric: metric function or None
        """
        self.target_metric = target_metric

    def _get_target_dtype(self):
        """
        Get appropriate dtype for target tensor based on criterion.

        Returns:
            torch dtype for target tensor
        """
        # CrossEntropyLoss and NLLLoss need long dtype (class indices)
        if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
            return torch.long
        else:
            return torch.float64

    def _prepare_target(self, output, batch_y):
        """
        Prepare target for loss computation.

        Args:
            output: model output
            batch_y: target batch

        Returns:
            prepared target tensor
        """
        # CrossEntropyLoss and NLLLoss expect [batch_size] shape
        if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
            return batch_y
        # Other losses may need reshaping
        else:
            return batch_y.reshape_as(output)

    def _compute_graph_loss(self, predictions: torch.Tensor,
                            batch_indices: np.ndarray) -> torch.Tensor:
        """
        Compute graph regularization loss with symmetric normalized Laplacian.

        L_sym = D^(-1/2) (D - W) D^(-1/2) = I - D^(-1/2) W D^(-1/2)

        Args:
            predictions: model predictions for batch [batch_size, output_dim]
            batch_indices: indices of batch elements in dataset

        Returns:
            loss: graph loss value
        """
        Y_batch = torch.tensor(self.Y_all[batch_indices], dtype=torch.float64).to(self.device)
        distances_sq = torch.cdist(Y_batch, Y_batch, p=2) ** 2
        nonzero_dists = distances_sq[distances_sq > 0]
        sigma_sq = self.sigma_sq_global

        W_batch = torch.exp(-distances_sq / (2 * sigma_sq))
        W_batch = W_batch - torch.diag(torch.diag(W_batch))
        D_diag = torch.sum(W_batch, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D_diag + 1e-10, -0.5))
        I = torch.eye(W_batch.shape[0], device=self.device, dtype=torch.float64)
        L_sym = I - D_inv_sqrt @ W_batch @ D_inv_sqrt
        loss = torch.trace(predictions.T @ L_sym @ predictions)
        n = predictions.shape[0]
        loss = loss / (n * (n-1))

        return loss


    def train(self, plot_convergence: bool = False, adaptive_lambda = False,
              early_stopping_patience: int = None, val_features: np.ndarray = None,
              val_target: np.ndarray = None, accuracy_check_interval: int = 10):
        """
        Train the model with combined loss (model loss + graph regularization loss).

        Args:
            plot_convergence: whether to plot convergence graphs after training
            adaptive_lambda: adaptive lambda method - False (disabled) or 'sobol' (once after 10% epochs)
            early_stopping_patience: number of epochs to wait for improvement before stopping (None = no early stopping)
            val_features: validation features for early stopping
            val_target: validation target for early stopping
            accuracy_check_interval: compute accuracy every N epochs (default: 10)

        Returns:
            self: trained model instance
        """
        self.model.train()
        model_losses = []
        graph_losses = []
        combined_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Early stopping setup
        best_val_loss = float('inf')
        prev_val_loss = float('inf')
        increasing_counter = 0  # Count consecutive epochs with increasing val loss
        self.best_model_state = None
        self.best_epoch = 0

        lam_nn = 1
        lam_graph = self.lambda_graph


        min_epochs_for_sobol = 30
        desired_sobol_epoch = int(self.num_epochs * 0.1)
        lmds_epochs = max(min_epochs_for_sobol, desired_sobol_epoch)

        if adaptive_lambda == 'sobol':
            if lmds_epochs >= self.num_epochs:
                if self.verbose:
                    print(f"  WARNING: Not enough epochs ({self.num_epochs}) for Sobol adaptation (requires {min_epochs_for_sobol})")
                    print(f"  Sobol will be skipped. Consider increasing num_epochs to at least {min_epochs_for_sobol * 2}")
            elif lmds_epochs > desired_sobol_epoch:
                if self.verbose:
                    print(f"  INFO: Sobol will be applied at epoch {lmds_epochs} (delayed from {desired_sobol_epoch} to ensure sufficient data)")

        lambdas_adapted = False

        lambda_history = {
            'initial_lambda_nn': lam_nn,
            'initial_lambda_graph': lam_graph,
            'changes': []  # List of {epoch, lambda_nn, lambda_graph, method}
        }

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.num_epochs}')

            indices = randperm(len(self.features)).numpy()

            epoch_model_losses = []
            epoch_graph_losses = []
            epoch_combined_losses = []

            # Training phase
            self.model.train()
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                batch_x = torch.tensor(self.features[batch_indices], dtype=fl64).to(self.device)

                # Get appropriate dtype for target based on criterion
                target_dtype = self._get_target_dtype()
                batch_y = torch.tensor(self.target[batch_indices], dtype=target_dtype).to(self.device)

                self.optimizer.zero_grad()

                output = self.model(batch_x)

                # Prepare target for loss computation
                prepared_target = self._prepare_target(output, batch_y)
                model_loss = self.criterion(output, prepared_target)

                graph_loss = self._compute_graph_loss(output, batch_indices)

                # Compute combined loss and backprop
                self.optimizer.zero_grad()
                combined_loss = lam_nn * model_loss + lam_graph * graph_loss
                combined_loss.backward()
                self.optimizer.step()

                epoch_model_losses.append(model_loss.item())
                epoch_graph_losses.append(graph_loss.item())
                epoch_combined_losses.append(combined_loss.item())

            model_losses.append(np.mean(epoch_model_losses))
            graph_losses.append(np.mean(epoch_graph_losses))
            combined_losses.append(np.mean(epoch_combined_losses))

            if epoch % accuracy_check_interval == 0 or epoch == self.num_epochs - 1:
                if self.task_type == 'classification':
                    self.model.eval()
                    with torch.no_grad():
                        # Train accuracy
                        train_pred = self.predict(self.features)
                        train_pred_classes = np.argmax(train_pred, axis=1)
                        train_acc = np.mean(train_pred_classes == self.target)
                        train_accuracies.append(train_acc)

                        if val_features is not None and val_target is not None:
                            val_pred = self.predict(val_features)
                            val_pred_classes = np.argmax(val_pred, axis=1)
                            val_acc = np.mean(val_pred_classes == val_target)
                            val_accuracies.append(val_acc)

            # Validation phase
            if val_features is not None and val_target is not None:
                self.model.eval()
                val_model_loss = self._compute_validation_loss(val_features, val_target)
                val_losses.append(val_model_loss)

                if self.verbose:
                    print(f'  Train - Model loss: {model_losses[-1]:.6f}, Graph loss: {graph_losses[-1]:.6f}, Combined: {combined_losses[-1]:.6f}')
                    print(f'  Val   - Model loss: {val_model_loss:.6f}')
                    if lambdas_adapted:
                        print(f'  Adaptive lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')
                    else:
                        print(f'  Lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')

                    if self.task_type == 'classification' and len(train_accuracies) > 0:
                        if epoch % accuracy_check_interval == 0 or epoch == self.num_epochs - 1:
                            print(f'  Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')

                # Early stopping check
                if early_stopping_patience is not None:
                    if val_model_loss < best_val_loss:
                        best_val_loss = val_model_loss
                        self.best_model_state = self.model.state_dict().copy()
                        self.best_epoch = epoch + 1
                        if self.verbose:
                            print(f'  New best model saved (val loss: {best_val_loss:.6f})')

                    # Check if val loss is increasing compared to best epoch
                    if val_model_loss >= best_val_loss:
                        increasing_counter += 1
                        if self.verbose:
                            print(f'  Val loss increasing: {increasing_counter}/{early_stopping_patience}')

                        if increasing_counter >= early_stopping_patience:
                            if self.verbose:
                                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                                print(f'Val loss increased for {early_stopping_patience} epochs')
                                print(f'Best model was at epoch {self.best_epoch} with val loss {best_val_loss:.6f}')
                            break
                    else:
                        increasing_counter = 0
                    prev_val_loss = val_model_loss
            else:
                if self.verbose:
                    print(f'  Model loss: {model_losses[-1]:.6f}, Graph loss: {graph_losses[-1]:.6f}, Combined: {combined_losses[-1]:.6f}')
                    if lambdas_adapted:
                        print(f'  Adaptive lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')
                    else:
                        print(f'  Lambdas: lam_nn={lam_nn:.6f}, lam_graph={lam_graph:.6f}')

            # Adaptive lambda update
            if adaptive_lambda == 'sobol' and epoch == lmds_epochs:
                # Sobol: compute once after 10% of epochs
                prev_lam_nn, prev_lam_graph = lam_nn, lam_graph
                lam_nn, lam_graph = _get_adaptive_lambda_sobol(combined_losses, model_losses, graph_losses)
                lambdas_adapted = True

                # Log lambda change
                lambda_history['changes'].append({
                    'epoch': epoch + 1,
                    'lambda_nn': float(lam_nn),
                    'lambda_graph': float(lam_graph),
                    'method': 'sobol',
                    'previous_lambda_nn': float(prev_lam_nn),
                    'previous_lambda_graph': float(prev_lam_graph)
                })

                if self.verbose:
                    print(f'  [Sobol] Lambda adapted at epoch {epoch + 1}')  

        self.trained_loss_values['model_loss'] = model_losses
        self.trained_loss_values['graph_loss'] = graph_losses
        self.trained_loss_values['combined_loss'] = combined_losses
        if val_features is not None:
            self.trained_loss_values['val_loss'] = val_losses

        if adaptive_lambda and len(lambda_history['changes']) > 0:
            self.trained_loss_values['lambda_history'] = lambda_history

        if self.task_type == 'classification':
            self.trained_loss_values['train_accuracy'] = train_accuracies
            self.trained_loss_values['val_accuracy'] = val_accuracies

        if plot_convergence:
            self._plot_convergence(combined_losses, None, model_losses, graph_losses,
                                  val_losses=val_losses if val_features is not None else None,
                                  best_epoch=self.best_epoch if hasattr(self, 'best_epoch') and self.best_epoch > 0 else None)

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

    def load_best_weights(self):
        """
        Load the best model weights saved during training (from early stopping).
        If no best weights were saved, keeps current weights.
        Also saves the best weights to disk.
        """
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Loaded best model weights from epoch {self.best_epoch}")

            if self.cache_folder is not None:
                best_weights_path = os.path.join(self.cache_folder, 'best_model.pth')
                self.save_weights(best_weights_path)
                if self.verbose:
                    print(f"Best model weights automatically saved to {best_weights_path}")
        else:
            if self.verbose:
                print("No best weights saved, using current model weights")

    def _compute_validation_loss(self, val_features: np.ndarray, val_target: np.ndarray) -> float:
        """
        Compute validation loss (model loss only, without graph regularization).

        Args:
            val_features: validation features [N_val, features]
            val_target: validation target [N_val, ] or [N_val, output_dim]

        Returns:
            val_loss: validation loss value
        """
        self.model.eval()
        with torch.no_grad():
            val_x = torch.tensor(val_features, dtype=fl64).to(self.device)

            target_dtype = self._get_target_dtype()
            val_y = torch.tensor(val_target, dtype=target_dtype).to(self.device)

            output = self.model(val_x)

            prepared_target = self._prepare_target(output, val_y)
            val_loss = self.criterion(output, prepared_target)

        return val_loss.item()

    def _plot_convergence(self, losses, lmds_epoch, nn_losses=None, graph_losses=None,
                          val_losses=None, best_epoch=None):
        """
        Plot training convergence graphs.

        Args:
            losses: combined loss values per epoch
            lmds_epoch: lambda values per epoch (not used in current implementation)
            nn_losses: model loss values per epoch
            graph_losses: graph loss values per epoch
            val_losses: validation loss values per epoch (optional)
            best_epoch: epoch with best validation loss (optional, 1-indexed)
        """
        if self.lambda_graph == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            epochs = range(1, len(losses) + 1)

            ax.plot(epochs, losses, label='Model Loss', color='blue', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Baseline Model Loss (no regularization)')
            ax.legend()
            ax.grid(True)
            ax.set_yscale('log')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            epochs = range(1, len(losses) + 1)

            axes[0].plot(epochs, losses, label='Combined Loss', color='blue')

            if val_losses is not None and best_epoch is not None and best_epoch <= len(losses):
                axes[0].axvline(x=best_epoch, color='gray', linestyle='--',
                               linewidth=1, alpha=0.7, label=f'Best Val Epoch ({best_epoch})')
                axes[0].axhline(y=losses[best_epoch-1], color='gray', linestyle='--',
                               linewidth=1, alpha=0.7)

            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Combined Loss')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_yscale('log')

            if nn_losses is not None and graph_losses is not None:
                axes[1].plot(epochs, nn_losses, label='Model Loss', color='green')
                axes[1].plot(epochs, graph_losses, label='Graph Loss', color='red')

                if val_losses is not None and best_epoch is not None and best_epoch <= len(nn_losses):
                    axes[1].axvline(x=best_epoch, color='gray', linestyle='--',
                                   linewidth=1, alpha=0.7, label=f'Best Val Epoch ({best_epoch})')
                    axes[1].axhline(y=nn_losses[best_epoch-1], color='green', linestyle='--',
                                   linewidth=1, alpha=0.5)

                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Model Loss vs Graph Loss')
                axes[1].legend()
                axes[1].grid(True)
                axes[1].set_yscale('log')

        plt.tight_layout()

        if self.cache_folder is not None:
            save_path = os.path.join(self.cache_folder, f"{self.model_name}_convergence.png")
            plt.savefig(save_path)
            if self.verbose:
                print(f"Convergence plot saved to {save_path}")

        plt.show()