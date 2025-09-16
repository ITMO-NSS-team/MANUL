import numpy as np
import torch
from torch import float32, nn
from sklearn.manifold import Isomap

class IntrinsicNN:
    def __init__(self, train_feature: np.ndarray,
                 train_target: np.ndarray,
                 latent_len: int,
                 num_epochs: int = 100,
                 ):
        self.trained_loss_values = {'model_loss': None}
        self.features = train_feature.astype(float)
        self.target = train_target
        self.num_epochs = num_epochs
        self.device = self.init_device()
        self._init_model(latent_len)

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

    def train(self, graph: np.ndarray):


        optim = torch.optim.AdamW(params=self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()


