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
                 epochs: int = 1000,
                 plot_convergence: bool = True,
                 checkpoint_each: [int, None] = 100,
                 save_checkpoint_history: bool = False,
                 logs_folder: [str, None] = None,
                 stop_criteria_value: float = 0.001
                 ):
        self.features = train_feature
        self.targets = train_target
        self.epochs = epochs
        self.plot_convergence = plot_convergence
        self.latent_len = latent_len
        self.checkpoint_each = checkpoint_each
        self.save_checkpoint_history = save_checkpoint_history
        self.stop_criteria_value = stop_criteria_value
        self.device = self._init_device()
        self.logs_folder = self._init_logs_folder(logs_folder)
        self.best_loss = np.inf
        self.best_isomap_model = None
        self.best_distances_matrix = None
        self.checkpoint_history_folder = None
        self.checkpoint_metadata = []

        if self.save_checkpoint_history and self.checkpoint_each is not None:
            self.checkpoint_history_folder = os.path.join(self.logs_folder, 'checkpoint_history')
            os.makedirs(self.checkpoint_history_folder, exist_ok=True)
            print(f'Checkpoint history enabled. Saving to: {self.checkpoint_history_folder}')

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
        return device

    def train(self):
        start_time = time.time()
        dist_train = self.generate_random_matrix(self.features.shape[0], dist_type='normal', device=self.device)
        isomap_model = IsomapNN(dist_train, n_components=self.latent_len, eigval_choice='MDS')
        isomap_model.to(self.device)
        isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.0001)
        isomap_criterion = nn.MSELoss()

        losses = []
        best_epoch = 0
        best_loss = np.inf
        epochs_list = []
        time_list = []

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
                # update best state
                best_epoch = epoch
                best_loss = losses[-1]
                self.best_isomap_model = isomap_model
                self.best_distances_matrix = self._isomap_weights(isomap_model)
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
                # Saving visualizations and logs each epoch checkpoint
                reproj_features = reproj_features.cpu().detach().numpy()
                output = output.cpu().detach().numpy()
                isomap_weights = self._isomap_weights(isomap_model)

                create_visualization(
                    epoch, losses, best_epoch, best_loss, best_reproj_features,
                    best_outputs, reproj_features,
                    output, self.targets,
                    isomap_weights,
                    isomap_eigenvalues[:3], self.logs_folder,
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
                np.save(f'{self.logs_folder}/best_mapping.npy', best_reproj_features)
                np.save(f'{self.logs_folder}/best_distance_matrix.npy', self.best_distances_matrix)
                print(f'Mapping saved: f"{self.logs_folder}/best_mapping.npy"'
                      f'\nDistances matrix saved:{self.logs_folder}/best_distance_matrix.npy')

                if self.save_checkpoint_history and self.checkpoint_history_folder is not None:
                    self._save_checkpoint_history(epoch, isomap_weights, losses[-1])

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

    def _save_checkpoint_history(self, epoch: int, distance_matrix: np.ndarray, loss: float):
        """
        Save distance matrix for current checkpoint to history folder.

        Args:
            epoch: Current epoch number
            distance_matrix: Distance matrix in upper triangular form (1D array)
            loss: Current loss value
        """
        checkpoint_filename = f'epoch_{epoch:05d}_distance_matrix.npy'
        checkpoint_path = os.path.join(self.checkpoint_history_folder, checkpoint_filename)

        np.save(checkpoint_path, distance_matrix)

        self.checkpoint_metadata.append({
            'epoch': int(epoch),
            'loss': float(loss),
            'filename': checkpoint_filename,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })

        metadata_path = os.path.join(self.checkpoint_history_folder, 'metadata.json')
        metadata = {
            'checkpoint_each': self.checkpoint_each,
            'total_epochs': self.epochs,
            'n_basis_points': self.features.shape[0],
            'checkpoints': self.checkpoint_metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if epoch % (self.checkpoint_each * 10) == 0:
            print(f'  Checkpoint history saved: epoch {epoch}')

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

    @staticmethod
    def load_checkpoint_history(checkpoint_history_folder: str):
        """
        Load all checkpoint distance matrices from history folder.

        Args:
            checkpoint_history_folder: Path to checkpoint_history folder

        Returns:
            dict with metadata and all distance matrices:
            {
                'metadata': {
                    'checkpoint_each': 100,
                    'total_epochs': 10000,
                    'n_basis_points': 1000
                },
                'checkpoints': [
                    {
                        'epoch': 0,
                        'loss': 0.5,
                        'timestamp': '...',
                        'distance_matrix': numpy array
                    },
                    ...
                ]
            }
        """
        metadata_path = os.path.join(checkpoint_history_folder, 'metadata.json')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        checkpoints = []
        for checkpoint_info in metadata['checkpoints']:
            checkpoint_path = os.path.join(checkpoint_history_folder, checkpoint_info['filename'])
            distance_matrix = np.load(checkpoint_path)

            checkpoints.append({
                'epoch': checkpoint_info['epoch'],
                'loss': checkpoint_info['loss'],
                'timestamp': checkpoint_info['timestamp'],
                'distance_matrix': distance_matrix
            })

        return {
            'metadata': {k: v for k, v in metadata.items() if k != 'checkpoints'},
            'checkpoints': checkpoints
        }
#todo: update readme with info about checpointing and adap lambdas options
