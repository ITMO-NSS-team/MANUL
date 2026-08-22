import os
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import float32, nn

from Adam.Isomap import IsomapNN
from Adam.visualization_utils import create_visualization, original_visualization_simple
from structure_approximation.IntrinsicNN import IntrinsicNN
from utils.manifold_diagnostics import (
    global_spectral_diagnostics,
    knn_indices_from_distance_matrix,
    local_curvature_proxies,
    correlate_local_curvature,
    neighbor_stability,
    local_triangle_violation_rate,
)


class GradientIsomap:
    def __init__(self, train_feature: torch.Tensor,
                 train_target: torch.Tensor,
                 latent_len: int,
                 n_neighbors: int = 25,
                 epochs: int = 1000,
                 plot_convergence: bool = True,
                 checkpoint_each: [int, None] = 100,
                 save_checkpoint_matrix: bool = False,
                 logs_folder: [str, None] = None,
                 stop_criteria_value: float = 0.001,
                 degenerate_kick_patience: int = 100,
                 degenerate_kick_noise_std: float = 0.005,
                 enable_kick: bool = False,
                 track_manifold_diagnostics: bool = False,
                 manifold_diag_local_k: int = 15,
                 manifold_diag_max_triples_per_point: int = 20,
                 analytic_curvature_ground_truth: [np.ndarray, None] = None,
                 manifold_diag_save_local_arrays: bool = False,
                 ):
        self.features = train_feature
        self.targets = train_target
        self.epochs = epochs
        self.plot_convergence = plot_convergence
        self.latent_len = latent_len
        self.n_neighbors = n_neighbors
        self.checkpoint_each = checkpoint_each
        self.save_checkpoint_matrix = save_checkpoint_matrix
        self.stop_criteria_value = stop_criteria_value
        # An occasional NaN/Inf gradient (see the guard in train()) is normal and
        # self-corrects on its own within a few epochs - it must not trigger any
        # response by itself. Only a genuinely SUSTAINED run of consecutive bad
        # gradients (degenerate_kick_patience epochs in a row) means the optimizer
        # is actually stuck rather than self-correcting, at which point the
        # distance matrix is directly perturbed with noise instead of waiting
        # indefinitely for a gradient signal that may never arrive.
        #
        # enable_kick defaults to OFF: on real MNIST-scale runs the kick itself
        # was observed to repeatedly destroy good convergence (best 0.0864 at
        # epoch 5131, then a series of kick-recover cycles landing at a worse
        # trough each time - never recovering by epoch 13300). It is kept only
        # as an optional, explicitly-opt-in mechanism; _consecutive_bad_grad_epochs
        # is still tracked and logged even when disabled, so the "would have
        # kicked" signal is visible in diagnostics without acting on it.
        self.degenerate_kick_patience = degenerate_kick_patience
        self.degenerate_kick_noise_std = degenerate_kick_noise_std
        self.enable_kick = enable_kick
        # Epoch-tracked invariant manifold diagnostics (global spectral shape
        # every epoch - free, reuses KernelPCA's already-computed full
        # spectrum; local curvature/stability/triangle-inequality proxies at
        # checkpoint intervals only, since those need a fresh local
        # Gram-eigendecomposition per point). See utils/manifold_diagnostics.py
        # and docs/regularization_investigation_journal.md for why these are
        # invariant/aggregate properties rather than raw value comparisons.
        self.track_manifold_diagnostics = track_manifold_diagnostics
        self.manifold_diag_local_k = manifold_diag_local_k
        self.manifold_diag_max_triples_per_point = manifold_diag_max_triples_per_point
        self.analytic_curvature_ground_truth = analytic_curvature_ground_truth
        # When True, dumps the PER-POINT local curvature proxy (not just its
        # mean) to disk at every checkpoint - needed for spatial ("how is
        # this proxy distributed over the surface, and does that pattern
        # change as training/W converges") visualizations, which the
        # scalar CSV columns alone cannot support.
        self.manifold_diag_save_local_arrays = manifold_diag_save_local_arrays
        self._prev_checkpoint_knn = None
        self._consecutive_bad_grad_epochs = 0
        self._init_device()
        self.logs_folder = self._init_logs_folder(logs_folder)
        self.best_loss = np.inf
        self.best_isomap_model = None
        self.best_distances_matrix = None
        self.checkpoint_history_folder = None
        self.checkpoint_metadata = []

        if self.save_checkpoint_matrix is not None:
            if self.checkpoint_each is None:
                print('To save distance matrices on checkpoints set "checkpoint_each" parameter differ from None')
            else:
                self.checkpoint_history_folder = os.path.join(self.logs_folder, 'checkpoints_history')
                os.makedirs(self.checkpoint_history_folder, exist_ok=True)
                print(f'Checkpoints history enabled. Saving to: {self.checkpoint_history_folder}')

    def _init_logs_folder(self, folder: [str, None]):
        if folder is None:
            logs_folder = f"gradisomap_{datetime.now().strftime('%d%m%Y-%H.%M')}"
        else:
            logs_folder = folder
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        print(f'Logs folder set as: {logs_folder}')
        return logs_folder

    def _init_device(self, device: str = None):
        """
        :param device: str - name of device
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        print(f'Device is {device}')
        self.device = device

    def train(self, use_init_assumption=False):
        start_time = time.time()
        if use_init_assumption:
            dist_train = torch.cdist(self.features, self.features)
        else:
            dist_train = self.generate_random_matrix(self.features.shape[0], dist_type='normal', device=self.device)
        isomap_model = IsomapNN(dist_train, n_components=self.latent_len, n_neighbors=self.n_neighbors, eigval_choice='MDS')
        isomap_model.to(self.device)
        isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.0001)
        isomap_criterion = nn.MSELoss()

        losses = []
        best_epoch = 0
        best_loss = np.inf
        epochs_list = []
        time_list = []
        best_reproj_features = None
        best_outputs = None

        # Per-epoch diagnostics (real history, unlike the old buggy Eigenvalues
        # column below which overwrote the whole CSV column with the latest
        # checkpoint's value every time). Meant to let a failure be diagnosed
        # after the fact instead of reacted to blindly with a kick: top
        # eigenvalue, the gap between the last KEPT eigenvalue and the first
        # EXCLUDED one (eigh's backward divides by exactly this kind of gap),
        # gradient health, and how long since the last improvement.
        diag_epochs = []
        diag_top_eigenvalue = []
        diag_boundary_gap = []
        diag_grad_norm = []
        diag_grad_nonfinite_count = []
        diag_had_bad_grad = []
        diag_lr = []
        diag_consecutive_bad_grad_epochs = []
        diag_would_kick = []
        diag_epochs_since_best = []
        eigenvalues_history = []

        # Group A manifold diagnostics (global spectral shape, every epoch).
        diag_lambda1 = []
        diag_lambda2 = []
        diag_lambda_ratio_12 = []
        diag_spectral_gap_12 = []
        diag_effective_rank = []
        diag_neg_mass_frac = []
        diag_top10_eigenvalues = []
        # Group C/D manifold diagnostics (local, checkpoint intervals only).
        local_diag_rows = []

        for epoch in range(self.epochs):
            reproj_features = isomap_model().to(float32)
            features = reproj_features.detach().clone()

            task_model = IntrinsicNN(features,
                                     self.targets,
                                     self.latent_len,
                                     plot_convergence=self.plot_convergence,
                                     epochs=500)
            task_model.train()

            output = task_model.model(reproj_features)
            isomap_loss = isomap_criterion(output.to(torch.float32),
                                           self.targets.reshape_as(output).to(torch.float32))
            losses.append(isomap_loss.item())

            if losses[-1] < best_loss or epoch == 0:
                best_epoch = epoch
                best_loss = losses[-1]
                self.best_isomap_model = isomap_model
                self.best_distances_matrix = isomap_model.distances_matrix
                best_reproj_features = reproj_features.cpu().detach().numpy()
                best_outputs = output.cpu().detach().numpy()

            isomap_optim.zero_grad(set_to_none=True)
            isomap_loss.backward()
            # eigh's backward divides by (eigenvalue_i - eigenvalue_j); near-degenerate
            # eigenvalues can turn that into NaN/Inf even when the forward loss is finite.
            # Zero those out instead of letting them corrupt the distance matrix permanently -
            # equivalent to skipping this epoch's update for the affected entries. Remember
            # whether this actually happened so _stable_eigenvalues reacts to a real bad
            # backward pass instead of guessing from the eigenvalue spectrum shape alone
            # (a close eigenvalue pair among the handful of kept components is common and
            # does not by itself mean the gradient was bad).
            had_bad_grad = False
            grad_nonfinite_count = 0
            for group in isomap_optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        nonfinite_mask = ~torch.isfinite(p.grad)
                        if nonfinite_mask.any():
                            had_bad_grad = True
                            grad_nonfinite_count += int(nonfinite_mask.sum().item())
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = float(isomap_model.layer.grad.norm().item()) if isomap_model.layer.grad is not None else float('nan')
            isomap_optim.step()
            print(f'epoch {epoch}/{self.epochs},  loss={losses[-1]}, lr={isomap_optim.param_groups[0]["lr"]}')

            isomap_eigenvalues = isomap_model.kernel_pca_.eigenvalues_.data.cpu().detach().numpy()
            would_kick = self._stable_eigenvalues(isomap_eigenvalues, isomap_optim, had_bad_grad, isomap_model,
                                                  current_loss=losses[-1], best_loss_so_far=best_loss)
            current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

            epochs_list.append(epoch)
            time_list.append(current_time)
            eigenvalues_history.append(','.join(str(float(x)) for x in isomap_eigenvalues[:3]))

            full_eigs = isomap_model.kernel_pca_.full_eigenvalues_sorted_
            boundary_gap = (abs(float(full_eigs[self.latent_len - 1]) - float(full_eigs[self.latent_len]))
                            if self.latent_len < len(full_eigs) else float('nan'))
            diag_epochs.append(epoch)
            diag_top_eigenvalue.append(float(isomap_eigenvalues[0]))
            diag_boundary_gap.append(boundary_gap)
            diag_grad_norm.append(grad_norm)
            diag_grad_nonfinite_count.append(grad_nonfinite_count)
            diag_had_bad_grad.append(had_bad_grad)
            diag_lr.append(isomap_optim.param_groups[0]['lr'])
            diag_consecutive_bad_grad_epochs.append(self._consecutive_bad_grad_epochs)
            diag_would_kick.append(would_kick)
            diag_epochs_since_best.append(epoch - best_epoch)

            if self.track_manifold_diagnostics:
                # Free: full_eigs is already computed above by KernelPCA's
                # fit; no extra eigendecomposition needed for Group A.
                spec_diag = global_spectral_diagnostics(full_eigs.cpu().numpy())
                diag_lambda1.append(spec_diag['lambda1'])
                diag_lambda2.append(spec_diag['lambda2'])
                diag_lambda_ratio_12.append(spec_diag['lambda_ratio_12'])
                diag_spectral_gap_12.append(spec_diag['spectral_gap'])
                diag_effective_rank.append(spec_diag['effective_rank'])
                diag_neg_mass_frac.append(spec_diag['neg_mass_frac'])
                diag_top10_eigenvalues.append(spec_diag['top_eigenvalues'])

            stop_criteria = self._check_stop_criteria(losses[-1])

            if (self.checkpoint_each is not None and epoch % self.checkpoint_each == 0 or
                    epoch == (self.epochs - 1) or stop_criteria):
                reproj_features = reproj_features.cpu().detach().numpy()
                output = output.cpu().detach().numpy()
                isomap_weights = self._isomap_weights(isomap_model)

                create_visualization(
                    epoch, losses, best_epoch, best_loss, best_reproj_features,
                    best_outputs, reproj_features,
                    output, self.targets,
                    isomap_weights,
                    isomap_eigenvalues[:3], self.checkpoint_history_folder,
                    current_time
                )

                df = pd.DataFrame(
                    columns=['Epochs', 'Time Spent', 'Loss', 'Eigenvalues'])
                df['Epochs'] = epochs_list
                df['Time Spent'] = time_list
                df['Loss'] = losses
                # Real per-epoch history (previously this assigned a single
                # joined string to the WHOLE column on every checkpoint,
                # silently overwriting all prior epochs' values with
                # whatever the latest checkpoint's top-3 eigenvalues were).
                df['Eigenvalues'] = eigenvalues_history
                df.to_csv(f'{self.logs_folder}/convergence_log.csv', index=False)

                diag_df = pd.DataFrame({
                    'Epochs': diag_epochs,
                    'Loss': losses,
                    'LR': diag_lr,
                    'TopEigenvalue': diag_top_eigenvalue,
                    'BoundaryGap': diag_boundary_gap,
                    'GradNorm': diag_grad_norm,
                    'GradNonfiniteCount': diag_grad_nonfinite_count,
                    'HadBadGrad': diag_had_bad_grad,
                    'ConsecutiveBadGradEpochs': diag_consecutive_bad_grad_epochs,
                    'WouldKick': diag_would_kick,
                    'EpochsSinceBest': diag_epochs_since_best,
                })
                diag_df.to_csv(f'{self.logs_folder}/diagnostics_log.csv', index=False)

                if self.track_manifold_diagnostics:
                    manifold_global_df = pd.DataFrame({
                        'Epochs': diag_epochs,
                        'Lambda1': diag_lambda1,
                        'Lambda2': diag_lambda2,
                        'LambdaRatio12': diag_lambda_ratio_12,
                        'SpectralGap12': diag_spectral_gap_12,
                        'EffectiveRank': diag_effective_rank,
                        'NegMassFrac': diag_neg_mass_frac,
                        'Top10Eigenvalues': diag_top10_eigenvalues,
                    })
                    manifold_global_df.to_csv(f'{self.logs_folder}/manifold_diagnostics_global.csv', index=False)

                    # Local diagnostics (Group C/D): more expensive (local
                    # k-NN Gram eigendecomposition per point), so only
                    # computed at checkpoint intervals, not every epoch.
                    dist_np = isomap_model.distances_matrix.detach().cpu().numpy()
                    knn = knn_indices_from_distance_matrix(dist_np, k=self.manifold_diag_local_k)
                    local_curv = local_curvature_proxies(dist_np, knn)

                    row = {
                        'Epoch': epoch,
                        'MeanLocalLambdaRatio': float(np.nanmean(local_curv['local_lambda_ratio'])),
                        'MeanLocalNegMassFrac': float(np.nanmean(local_curv['local_neg_mass_frac'])),
                    }

                    if self.analytic_curvature_ground_truth is not None:
                        corr = correlate_local_curvature(
                            local_curv['local_lambda_ratio'], self.analytic_curvature_ground_truth)
                        row['CurvatureCorrPearson'] = corr['pearson_r']
                        row['CurvatureCorrSpearman'] = corr['spearman_r']

                    if self._prev_checkpoint_knn is not None:
                        stability = neighbor_stability(self._prev_checkpoint_knn, knn)
                        row['MeanNeighborJaccardVsPrevCheckpoint'] = stability['mean_jaccard']
                    else:
                        row['MeanNeighborJaccardVsPrevCheckpoint'] = float('nan')
                    self._prev_checkpoint_knn = knn

                    triangle = local_triangle_violation_rate(
                        dist_np, knn, max_triples_per_point=self.manifold_diag_max_triples_per_point)
                    row['LocalTriangleViolationRate'] = triangle['violation_rate']

                    local_diag_rows.append(row)
                    pd.DataFrame(local_diag_rows).to_csv(
                        f'{self.logs_folder}/manifold_diagnostics_local.csv', index=False)

                    if self.manifold_diag_save_local_arrays:
                        arrays_folder = os.path.join(self.logs_folder, 'manifold_diag_local_arrays')
                        os.makedirs(arrays_folder, exist_ok=True)
                        np.save(os.path.join(arrays_folder, f'epoch_{epoch}_local_lambda_ratio.npy'),
                                local_curv['local_lambda_ratio'])
                        np.save(os.path.join(arrays_folder, f'epoch_{epoch}_local_neg_mass_frac.npy'),
                                local_curv['local_neg_mass_frac'])

                torch.save(self.best_isomap_model.state_dict(), f'{self.logs_folder}/best_isomap_model.pt')
                np.save(f'{self.logs_folder}/best_distance_matrix.npy', self.best_distances_matrix.detach().cpu().numpy())
                print(f'Distances matrix saved:{self.logs_folder}/best_distance_matrix.npy')

                if self.save_checkpoint_matrix:
                    self._save_checkpoint_weights_matrix(epoch, isomap_weights)

            if stop_criteria:
                break

        torch.save(self.best_isomap_model.state_dict(), f'{self.logs_folder}/best_isomap_model.pt')
        print(f'Train finished in {time_list[-1]}, logs folder: {self.logs_folder}')

    def visualize_trained(self):
        proj_features = self.best_isomap_model().to(float32)
        features = proj_features.detach().clone()
        task_model = IntrinsicNN(features,
                                 self.targets,
                                 self.latent_len,
                                 plot_convergence=self.plot_convergence,
                                 epochs=500)
        task_model.train()
        output = task_model.model(features).flatten().cpu().detach().numpy()
        original_visualization_simple(self.features.cpu().detach().numpy(),
                                      self.targets.cpu().detach().numpy(),
                                      output,
                                      save_path=f'{self.logs_folder}/prediction_train.png')

    def _save_checkpoint_weights_matrix(self, epoch: int, distance_matrix: np.ndarray):
        """
        Save distance matrix for current checkpoint to history folder.

        Args:
            epoch: Current epoch number
            distance_matrix: Distance matrix in upper triangular form (1D array)
        """
        checkpoint_filename = f'{epoch}_epoch_distance_matrix.npy'
        checkpoint_path = os.path.join(self.checkpoint_history_folder, checkpoint_filename)

        np.save(checkpoint_path, distance_matrix)

    def _check_stop_criteria(self, loss_value: float):
        return loss_value <= self.stop_criteria_value

    def _isomap_weights(self, isomap_model):
        weights_matrix = isomap_model.distances_matrix.cpu().detach().clone().numpy()
        rows, cols = weights_matrix.shape
        upper_tri_indices = torch.triu_indices(rows, cols, offset=1)
        isomap_weights = weights_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        return isomap_weights.cpu().detach().numpy() if hasattr(isomap_weights, 'cpu') else isomap_weights

    def _stable_eigenvalues(self, isomap_eigenvalues, isomap_optim, had_bad_grad=False, isomap_model=None,
                           current_loss=None, best_loss_so_far=None):
        # LR switching is back to the original, long-proven check (top eigenvalue
        # magnitude only). An occasional NaN/Inf gradient (see train()'s guard) is
        # normal and self-corrects on its own within a few epochs - it must NOT
        # bump the LR on every single occurrence, that overreacts to a non-problem
        # and was observed to stall convergence instead of protecting it.
        degenerate = abs(isomap_eigenvalues[0]) < 0.01
        for param_group in isomap_optim.param_groups:
            param_group['lr'] = 0.01 if degenerate else 0.0001
        if degenerate:
            print('Egv degenerate')

        # Separate, much coarser safety net: only a genuinely SUSTAINED run of
        # consecutive bad-gradient epochs (not an occasional one-off) means the
        # optimizer is actually stuck rather than self-correcting - only then is
        # a direct perturbation of the distance matrix warranted.
        if had_bad_grad:
            self._consecutive_bad_grad_epochs += 1
        else:
            self._consecutive_bad_grad_epochs = 0

        # A SHARP, well-converged optimum barely moves epoch to epoch, so it can
        # produce the exact same near-degenerate eigenvalue spectrum (and hence a
        # bad gradient) every single epoch - which looks identical to "stuck in a
        # bad state" by consecutive-epoch count alone. Diagnosed on real MNIST-scale
        # data: the model converged to loss~2.64 (best-ever was 2.6155) and held
        # there for 100+ epochs, and the kick fired anyway and destroyed it -
        # after which the run never recovered a loss below ~4 for the remaining
        # ~19000 epochs. If the current loss is already at/near the best ever
        # found, this is convergence, not stuck-badness, and must NOT be kicked.
        near_best = (current_loss is not None and best_loss_so_far is not None
                    and np.isfinite(best_loss_so_far)
                    and current_loss <= best_loss_so_far * 1.1)
        if near_best:
            self._consecutive_bad_grad_epochs = 0

        would_kick = self._consecutive_bad_grad_epochs >= self.degenerate_kick_patience
        if would_kick and isomap_model is not None:
            if self.enable_kick:
                with torch.no_grad():
                    isomap_model.layer.add_(
                        torch.randn_like(isomap_model.layer) * self.degenerate_kick_noise_std)
                print(f'Kicked distance matrix after {self._consecutive_bad_grad_epochs} '
                      f'consecutive bad-gradient epochs')
            else:
                print(f'Kick condition met after {self._consecutive_bad_grad_epochs} '
                      f'consecutive bad-gradient epochs (enable_kick=False, not acting - diagnostics only)')
            self._consecutive_bad_grad_epochs = 0
        return would_kick

    @staticmethod
    def generate_random_matrix(n_samples, dist_type='normal', device='cuda'):
        if dist_type == 'uniform':
            matrix = torch.rand(n_samples, n_samples, device=device)
        elif dist_type == 'normal':
            matrix = torch.randn(n_samples, n_samples, device=device).abs()
        elif dist_type == 'exp':
            matrix = torch.rand(n_samples, n_samples, device=device).pow(2)

        matrix = (matrix + matrix.T) / 2
        matrix.fill_diagonal_(0)
        return matrix / matrix.max()

