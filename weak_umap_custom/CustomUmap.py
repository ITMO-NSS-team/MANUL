import torch
from torch import nn
import matplotlib.pyplot as plt

device = 'cuda'


class CustomUmap(nn.Module):
    def __init__(self, points_initial_assumption: torch.tensor):
        super().__init__()
        self.embedding = points_initial_assumption
        self.params = torch.nn.Parameter(points_initial_assumption)

    def forward(self):
        self.embedding = torch.tensor(self.params)
        return self.embedding
