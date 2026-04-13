import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import float32, nn


class IntrinsicNN:
    def __init__(self, train_features: torch.Tensor,
                 train_targets: torch.Tensor,
                 latent_len: int,
                 output_dim: int = None,
                 epochs: int = 150,
                 plot_convergence: bool = True
                 ):
        """
        Args:
            train_features: [N, latent_len] Isomap coordinates
            train_targets: [N] scalar or [N, C] multi-dim targets
            latent_len: intrinsic dimensionality
            output_dim: output dimension (None = infer from target shape)
            epochs: training epochs
            plot_convergence: show convergence plot
        """
        self.features = train_features
        self.targets = train_targets
        self.epochs = epochs
        self.plot_convergence = plot_convergence
        self.device = self.init_device()
        self.latent_len = latent_len
        self.convergence_history = None
        self.loss = None

        # Infer output_dim from target shape
        if output_dim is None:
            if train_targets.dim() == 1 or (train_targets.dim() == 2 and train_targets.shape[1] == 1):
                self.output_dim = 1
            else:
                self.output_dim = train_targets.shape[1]
        else:
            self.output_dim = output_dim

        self._init_model(latent_len, self.output_dim)

    def init_device(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device

    def _init_model(self, latent_len: int, output_dim: int):
        """
        Compact NN with fixed degrees of freedom.
        Original: Linear(latent_len, latent_len) → Linear(latent_len, 1)
        Multi-dim: Linear(latent_len, latent_len) → Linear(latent_len, output_dim)
        """
        seq = [nn.Linear(latent_len, latent_len, dtype=float32),
               nn.Linear(latent_len, output_dim, dtype=float32)]
        self.model = nn.Sequential(*seq).to(self.device)

    def _plot_convergence(self):
        if self.convergence_history is not None:
            plt.plot(np.arange(len(self.convergence_history)), self.convergence_history)
            plt.title(f'Convergence, latent_dim={self.latent_len}, output_dim={self.output_dim}')
            plt.ylabel('MSE loss')
            plt.xlabel('Epoch')
            plt.show()
        else:
            raise Warning('Can not visualize convergence plot as history is None')

    def _check_stop_criteria(self, losses_list, no_changes: int = 50):
        if len(losses_list) >= no_changes:
            if abs(losses_list[-1] - losses_list[-no_changes]) < 0.001:
                return True
        else:
            return False

    def train(self):
        """Training loop for manifold approximation."""
        self.features = self.features.to(torch.float32).to(self.device)
        self.targets = self.targets.to(torch.float32).to(self.device)

        optim = torch.optim.AdamW(params=self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for ep in range(self.epochs):
            optim.zero_grad()
            out = self.model(self.features)
            # reshape_as handles both:
            # scalar: targets [N] → reshape to [N,1] to match out [N,1]
            # multi:  targets [N,C] → no-op, matches out [N,C]
            task_loss = criterion(out, self.targets.reshape_as(out))
            losses.append(task_loss.item())
            task_loss.backward()
            optim.step()
            if self._check_stop_criteria(losses):
                break

        self.loss = losses[-1]
        self.convergence_history = losses
        if self.plot_convergence:
            self._plot_convergence()