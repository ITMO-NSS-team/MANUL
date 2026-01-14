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
                 stop_criteria_value: float = 0.001
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
            isomap_optim.step()
            print(f'epoch {epoch}/{self.epochs},  loss={losses[-1]}, lr={isomap_optim.param_groups[0]["lr"]}')

            isomap_eigenvalues = isomap_model.kernel_pca_.eigenvalues_.data.cpu().detach().numpy()
            self._stable_eigenvalues(isomap_eigenvalues, isomap_optim)
            current_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

            epochs_list.append(epoch)
            time_list.append(current_time)

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
                df['Eigenvalues'] = ','.join(str(float(x)) for x in isomap_eigenvalues[:3])
                df.to_csv(f'{self.logs_folder}/convergence_log.csv', index=False)

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

    def _stable_eigenvalues(self, isomap_eigenvalues, isomap_optim):
        if abs(isomap_eigenvalues[0]) < 0.01:
            for param_group in isomap_optim.param_groups:
                param_group['lr'] = 0.01
                print('Egv degenerate')
        if abs(isomap_eigenvalues[0]) >= 0.01:
            for param_group in isomap_optim.param_groups:
                param_group['lr'] = 0.0001

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

