import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import float32, nn


class IntrinsicNN:
    def __init__(self, train_features: torch.Tensor,
                 train_targets: torch.Tensor,
                 latent_len: int,
                 epochs: int = 150,
                 plot_convergence: bool = True
                 ):

        self.features = train_features
        self.targets = train_targets
        self.epochs = epochs
        self.plot_convergence = plot_convergence
        self.device = self.init_device()
        self.latent_len = latent_len
        self._init_model(latent_len)
        self.convergence_history = None
        self.loss = None

    def init_device(self, device: str = None):
        """
        :param device: str - name of device
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device

    def _init_model(self, latent_len: int):
        """
        Function for initialization of linear neural network with fixed degrees of freedom (latent length)
        """
        seq = [nn.Linear(latent_len, latent_len, dtype=float32),
               nn.Linear(latent_len, 1, dtype=float32)]
        self.model = nn.Sequential(*seq).to(self.device)

    def _plot_convergence(self):
        if self.convergence_history is not None:
            plt.plot(np.arange(len(self.convergence_history)), self.convergence_history)
            plt.title(f'Convergence plot, latent_dim={self.latent_len}')
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
        """
        Function for training loop with compact NN for manifold approximation
        """

        optim = torch.optim.AdamW(params=self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for ep in range(self.epochs):
            optim.zero_grad()
            out = self.model(self.features)
            task_loss = criterion(out, self.targets.reshape_as(out))
            # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
            losses.append(task_loss.item())
            task_loss.backward()
            optim.step()
            if self._check_stop_criteria(losses):
                break

        self.loss = losses[-1]

        self.convergence_history = losses
        if self.plot_convergence:
            self._plot_convergence()
