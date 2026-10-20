import copy
import os
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    sampling_D = 2  # as combine 2 features
    group_size = sampling_D * 2 + 2  # required sample multiple for calc_second_order=True
    n_samples = len(combines_loss) // group_size  # use as much of the given window as fits

    if n_samples < 1:
        print('Epochs number is too small to calculate adaptive lambda')
        return [1, 1]

    combines_loss = np.array(combines_loss)
    nn_loss = np.expand_dims(np.array(nn_loss), axis=1)
    graph_loss = np.expand_dims(np.array(graph_loss), axis=1)

    X_array = np.hstack((nn_loss, graph_loss))

    bounds = [[-100, 100] for i in range(sampling_D)]
    names = ['x{}'.format(i) for i in range(sampling_D)]

    n_used = n_samples * group_size
    X_array = X_array[:n_used]
    combines_loss = combines_loss[:n_used]

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

        # RBF bandwidth for the A_ij = exp(-W_ij^2 / sigma_sq) affinity used
        # by _compute_graph_loss_global* (calibrate_rbf_bandwidth=True).
        # Median-based, same approach compute_full_rbf_affinity already uses
        # elsewhere in this file (that function is otherwise unused). Without
        # this, exp(-W_ij^2) implicitly assumes W spans well below/above 1 -
        # but the learned manifold distances here were found to sit in
        # roughly [0, 1], so the raw kernel barely discriminates near vs far
        # (empirically: A ranged 0.37-1.0, std 0.06 - close to a constant).
        offdiag_mask = ~np.eye(self.weights_matrix.shape[0], dtype=bool)
        median_dist = np.median(self.weights_matrix[offdiag_mask]) if self.weights_matrix.shape[0] > 1 else 0.0
        self.rbf_sigma_sq = float(median_dist ** 2) if median_dist > 1e-12 else 1.0
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

    def _compute_graph_loss_global(self, all_predictions: torch.Tensor, batch_indices=None,
                                   sigma_sq: float = 1.0) -> torch.Tensor:
        """
        Compute global graph regularization loss using pairwise distances.

        Computes weighted sum of prediction differences across all point pairs:
        Loss = (1/2N^2) * sum_ij (A_ij * ||f(i) - f(j)||^2)
        where A=exp(-W**2/sigma_sq) and W is the precomputed distance matrix and f(i) are model predictions.
        Args:
            all_predictions: model predictions for all points [N, output_dim]
            batch_indices: indices of the current batch (optional). If provided,
                           computes loss only for pairs within the batch.
            sigma_sq: RBF bandwidth (default 1.0 = original, uncalibrated
                behavior). Pass self.rbf_sigma_sq (median-distance-based) to
                calibrate the kernel to the manifold's actual distance scale -
                see the calibrate_rbf_bandwidth note in train().
        Returns:
            graph_loss: scalar regularization loss tensor (kept on the autograd
                graph - see note below on why this matters)
        """
        # F used to be detached to numpy here, and prediction_dists/graph_loss
        # computed entirely with sklearn/numpy - meaning graph_loss carried no
        # gradient at all. combined_loss = lam_nn*model_loss + lam_graph*graph_loss
        # still looked like a normal tensor (adding a python/numpy scalar to a
        # tensor is valid), so combined_loss.backward() ran without error, but
        # the constant graph_loss term has zero derivative - regularization
        # never affected a single weight update, on either the synthetic or
        # MNIST pipelines, regardless of lam_graph's value. Verified directly:
        # gradients were bit-for-bit identical with and without the graph term.
        # torch.cdist keeps this differentiable so graph_loss actually
        # constrains predictions of nearby (per the learned manifold) points.
        F = all_predictions
        if batch_indices is not None:
            # loss calculates only on graph base points to save weights_matrix dimensionality
            real_indices_in_base = np.intersect1d(batch_indices,
                                                  self.base_indices)  # find batch indices which are in base
            if len(real_indices_in_base) == 0:
                # This batch has no overlap with the manifold's landmark
                # points - nothing to regularize against. Returning 0 (rather
                # than the previous code's implicit 0/0) avoids a NaN that
                # would now propagate into the gradient, since graph_loss is
                # no longer detached.
                return torch.zeros((), dtype=F.dtype, device=F.device)
            indices_in_batch_indices = np.argwhere(np.isin(batch_indices, real_indices_in_base))[:, 0]
            F = F[indices_in_batch_indices]
            prediction_dists = torch.cdist(F, F, p=2) ** 2

            # find valid indices to cut weights_matrix with base dimensionality
            indices_in_base = np.argwhere(np.isin(self.base_indices, real_indices_in_base))[:, 0]
            batch_weights_matrix = self.weights_matrix[indices_in_base][:, indices_in_base]
            batch_weights_matrix_t = torch.as_tensor(batch_weights_matrix, dtype=F.dtype, device=F.device)
            graph_loss = torch.sum(prediction_dists * torch.exp(-batch_weights_matrix_t ** 2 / sigma_sq)) / (
                        2 * len(real_indices_in_base) ** 2)
        else:
            prediction_dists = torch.cdist(F, F, p=2) ** 2
            weights_matrix_t = torch.as_tensor(self.weights_matrix, dtype=F.dtype, device=F.device)
            graph_loss = torch.sum(prediction_dists * torch.exp(-weights_matrix_t ** 2 / sigma_sq)) / (2 * F.shape[0] ** 2)
        return graph_loss

    def _compute_graph_loss_global_normalized(self, all_predictions: torch.Tensor, batch_indices=None,
                                              eps: float = 1e-6, sigma_sq: float = 1.0) -> torch.Tensor:
        """
        Same weighted-pairwise-distance graph loss as _compute_graph_loss_global,
        but z-scores predictions (over the same point set the pairwise distances
        are computed on) before computing distances.

        Without this, graph_loss can be minimized towards exactly 0 by
        collapsing ALL predictions to a single constant, regardless of the
        A_ij weighting: ||f(i)-f(j)||^2 = 0 for every pair (near or far) at
        once, since the loss only penalizes dissimilarity of nearby pairs and
        never penalizes similarity - there is nothing pushing distant pairs
        apart. Normalizing first makes graph_loss invariant to the overall
        scale/shift of predictions, so uniformly shrinking them towards a
        constant no longer reduces it - only actually rearranging predictions
        relative to each other does.

        eps guards std -> 0 early in training (e.g. right after init, before
        the model has learned to produce varied outputs): with eps in the
        denominator (rather than a floor on std itself), if the true spread
        is much smaller than eps, normalization is effectively a no-op and
        graph_loss stays small (matching the unnormalized version's
        behavior) instead of amplifying near-zero differences into large,
        noisy values - it only starts actively normalizing once predictions
        have enough genuine spread to matter.

        sigma_sq: RBF bandwidth (default 1.0 = original, uncalibrated
            behavior). Pass self.rbf_sigma_sq to calibrate - see the
            calibrate_rbf_bandwidth note in train().
        """
        F = all_predictions
        if batch_indices is not None:
            real_indices_in_base = np.intersect1d(batch_indices, self.base_indices)
            if len(real_indices_in_base) == 0:
                return torch.zeros((), dtype=F.dtype, device=F.device)
            indices_in_batch_indices = np.argwhere(np.isin(batch_indices, real_indices_in_base))[:, 0]
            F = F[indices_in_batch_indices]
            F = (F - F.mean()) / (F.std(unbiased=False) + eps)
            prediction_dists = torch.cdist(F, F, p=2) ** 2

            indices_in_base = np.argwhere(np.isin(self.base_indices, real_indices_in_base))[:, 0]
            batch_weights_matrix = self.weights_matrix[indices_in_base][:, indices_in_base]
            batch_weights_matrix_t = torch.as_tensor(batch_weights_matrix, dtype=F.dtype, device=F.device)
            graph_loss = torch.sum(prediction_dists * torch.exp(-batch_weights_matrix_t ** 2 / sigma_sq)) / (
                        2 * len(real_indices_in_base) ** 2)
        else:
            F = (F - F.mean()) / (F.std(unbiased=False) + eps)
            prediction_dists = torch.cdist(F, F, p=2) ** 2
            weights_matrix_t = torch.as_tensor(self.weights_matrix, dtype=F.dtype, device=F.device)
            graph_loss = torch.sum(prediction_dists * torch.exp(-weights_matrix_t ** 2 / sigma_sq)) / (2 * F.shape[0] ** 2)
        return graph_loss

    def train(self, plot_convergence: bool = False, adaptive_lambda=False,
              early_stopping_patience: int = 100, adaptive_lambda_window: int = 100,
              adaptive_lambda_recompute: bool = False, normalize_graph_loss: bool = False,
              calibrate_rbf_bandwidth: bool = False):
        """
        Train the model with combined loss (model loss + graph regularization loss).

        Args:
            plot_convergence: whether to plot convergence graphs after training
            adaptive_lambda: adaptive lambda method - False (disabled) or 'sobol'
            early_stopping_patience: number of epochs to wait for improvement before stopping (None = no early stopping)
            adaptive_lambda_window: size (in epochs) of the loss history window the Sobol analysis
                is run on. With adaptive_lambda_recompute=False this is also the single epoch at
                which lambdas are computed once and then kept fixed for the rest of training. With
                adaptive_lambda_recompute=True it is also the recompute period (lambdas are
                recalculated every adaptive_lambda_window epochs, each time from the most recent
                window of loss history) - this makes recomputation cheap since Sobol analysis only
                runs once per window instead of every epoch.
            adaptive_lambda_recompute: if True, keep recalculating lambdas every
                adaptive_lambda_window epochs for the whole training run instead of only once.
            normalize_graph_loss: if True, use _compute_graph_loss_global_normalized
                (z-scores predictions before computing pairwise distances, so
                collapsing predictions towards a constant no longer trivially
                shrinks graph_loss) instead of the original
                _compute_graph_loss_global. Defaults to False so existing
                behavior/callers are unaffected.
            calibrate_rbf_bandwidth: if True, use self.rbf_sigma_sq (median
                squared distance over the manifold's landmark-to-landmark
                distance matrix) as the RBF kernel's bandwidth in
                A_ij = exp(-W_ij^2 / sigma_sq), instead of the uncalibrated
                A_ij = exp(-W_ij^2). Found empirically (2026-08) that without
                this, when the learned distances sit roughly in [0, 1] (as
                they did on synthetic data), the raw kernel barely
                discriminates near vs far pairs (A ranged 0.37-1.0, std 0.06
                - close to a constant weight on every pair) - graph_loss then
                mostly just pulls all landmark predictions towards a common
                value rather than encoding real local manifold structure.
                Defaults to False so existing behavior/callers are unaffected.
        Returns:
            self: trained model instance
        """
        self.model.train()

        best_val_loss = float('inf')
        increasing_counter = 0

        lam_nn = 1
        lam_graph = 1

        no_changes_epochs = 100
        sigma_sq = self.rbf_sigma_sq if calibrate_rbf_bandwidth else 1.0

        if adaptive_lambda == 'sobol':
            if adaptive_lambda_window >= self.num_epochs:
                print(
                    f"Please use at least {adaptive_lambda_window} number of epochs of set adaptive_lambda as 'False'")

        start_time = time.time()
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')

            self.model.train()

            # Fixed order (not shuffled) - matches baseline_train_test's
            # DataLoader(shuffle=False) so both pipelines process data the
            # same way and any quality difference reflects the regularization
            # itself, not an unrelated pipeline discrepancy. These must stay
            # real dataset indices (not batch-local positions), since
            # _compute_graph_loss_global intersects batch_indices against
            # self.base_indices (the manifold's landmark points).
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
                if normalize_graph_loss:
                    graph_loss = self._compute_graph_loss_global_normalized(output, batch_indices=batch_indices,
                                                                            sigma_sq=sigma_sq)
                else:
                    graph_loss = self._compute_graph_loss_global(output, batch_indices=batch_indices,
                                                                 sigma_sq=sigma_sq)

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
                        # self.model is a live reference that optimizer.step()
                        # keeps mutating in place - assigning it directly here
                        # (as opposed to a real snapshot) means self.best_model
                        # would silently end up identical to the FINAL epoch's
                        # model by the time training exits, defeating the
                        # entire point of tracking a best epoch.
                        self.best_model = copy.deepcopy(self.model)
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
                                self.convergence_history['model_lambda'].append(float(lam_nn))
                                self.convergence_history['graph_lambda'].append(float(lam_graph))
                                break

            if adaptive_lambda == 'sobol':
                if epoch < adaptive_lambda_window:
                    self.convergence_history['model_lambda'].append(1.0)
                    self.convergence_history['graph_lambda'].append(1.0)
                elif epoch == adaptive_lambda_window or (
                        adaptive_lambda_recompute and
                        (epoch - adaptive_lambda_window) % adaptive_lambda_window == 0):
                    # Recompute from the most recent window of loss history - on the first
                    # trigger (epoch == adaptive_lambda_window) that is the whole window;
                    # on later periodic triggers it is the window since the previous recompute.
                    window_start = epoch - adaptive_lambda_window + 1
                    lam_nn, lam_graph = get_adaptive_lambda_sobol(
                        self.convergence_history['combined_loss'][window_start:epoch + 1],
                        self.convergence_history['model_loss'][window_start:epoch + 1],
                        self.convergence_history['graph_loss'][window_start:epoch + 1])
                    self.convergence_history['model_lambda'].append(float(lam_nn))
                    self.convergence_history['graph_lambda'].append(float(lam_graph))
                else:
                    self.convergence_history['model_lambda'].append(float(lam_nn))
                    self.convergence_history['graph_lambda'].append(float(lam_graph))

        self.convergence_history['epoch'] = np.arange(1, len(self.convergence_history['model_loss']) + 1)
        # Must match the actual condition used in the epoch loop above
        # (`if adaptive_lambda == 'sobol':`) - the loop only ever appends to
        # model_lambda/graph_lambda under that exact condition, leaving them
        # empty for ANY other value (False, None, or anything else),
        # including train()'s own default (adaptive_lambda=False). The old
        # `if adaptive_lambda is None:` check missed the False case - the
        # DEFAULT parameter value - so calling train() with adaptive_lambda
        # left unset (or explicitly False) crashed here on a length mismatch
        # between the populated *_loss lists (len == epochs run) and the
        # still-empty model_lambda/graph_lambda lists (len == 0).
        if adaptive_lambda != 'sobol':
            n_completed_epochs = len(self.convergence_history['model_loss'])
            self.convergence_history['model_lambda'] = np.ones(n_completed_epochs)
            self.convergence_history['graph_lambda'] = np.ones(n_completed_epochs)
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
        combined_loss = self.convergence_history['combined_loss']
        model_loss = self.convergence_history['model_loss']
        graph_loss = self.convergence_history['graph_loss']
        val_loss = self.convergence_history['val_loss']
        model_lambda = self.convergence_history['model_lambda']
        graph_lambda = self.convergence_history['graph_lambda']

        fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(combined_loss) + 1)
        axes[0].plot(epochs, combined_loss, label='Combined Loss', color='blue', linewidth=2)
        if len(val_loss) != 0 and self.best_epoch is not None and self.best_epoch <= len(combined_loss):
            axes[0].axvline(x=self.best_epoch, color='gray', linestyle='--',
                            linewidth=1.5, alpha=0.7, label=f'Best Val Epoch ({self.best_epoch})')
            axes[0].axhline(y=combined_loss[self.best_epoch - 1], color='gray', linestyle='--',
                            linewidth=1, alpha=0.5)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Plot 2: Model Loss vs Graph Loss
        if len(model_loss) != 0 and len(graph_loss) != 0:
            axes[1].plot(epochs, np.array(model_loss)*np.array(model_lambda), label='Model Loss', color='green', linewidth=2)
            axes[1].plot(epochs, np.array(graph_loss)*np.array(graph_lambda), label='Graph Loss', color='red', linewidth=2)

            if len(val_loss) != 0 and self.best_epoch != 0 and self.best_epoch <= len(model_loss):
                axes[1].axvline(x=self.best_epoch, color='gray', linestyle='--',
                                linewidth=1.5, alpha=0.7, label=f'Best Val Epoch ({self.best_epoch})')
                axes[1].axhline(y=model_loss[self.best_epoch - 1], color='green', linestyle='--',
                                linewidth=1, alpha=0.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Model Loss vs Graph Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            #axes[1].set_yscale('log')
        plt.tight_layout()

        if self.cache_folder is not None:
            save_path = os.path.join(self.cache_folder, "Loss_Components_Convergence.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()

        if len(val_loss) > 0:
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, model_loss, label='Training Model Loss',
                    color='blue', linewidth=2, alpha=0.8)
            val_epochs = range(1, len(val_loss) + 1)
            ax.plot(val_epochs, val_loss, label='Validation Loss',
                    color='orange', linewidth=2, alpha=0.9, linestyle='-', markersize=4)
            if self.best_epoch is not None and self.best_epoch <= len(model_loss):
                ax.axvline(x=self.best_epoch, color='red', linestyle='--',
                           linewidth=2, alpha=0.8, label=f'Best Epoch ({self.best_epoch})')
                if self.best_epoch <= len(val_loss):
                    best_val_loss = val_loss[self.best_epoch - 1]
                else:
                    best_val_loss = val_loss[-1] if val_loss else 0
                ax.plot(self.best_epoch, best_val_loss, 'r', markersize=5,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
                        label=f'Best Val Loss: {best_val_loss:.4f}')
                ax.axhline(y=best_val_loss, color='red', linestyle=':',
                           linewidth=1, alpha=0.5)
            train_info = f'Training epochs: {len(combined_loss)}\n'
            if self.best_epoch is not None:
                train_info += f'Best epoch: {self.best_epoch}\n'
                train_info += f'Final train loss: {model_loss[-1]:.6f}\n'
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
            ax.set_xlim(0, max(len(combined_loss), len(val_loss)) + 1)
            plt.tight_layout()

            if self.cache_folder is not None:
                save_path = os.path.join(self.cache_folder, "Train_Validation_Convergence.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print(f"Train vs Validation convergence plot saved to {save_path}")
            else:
                plt.show()
