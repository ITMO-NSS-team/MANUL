from torch import nn, float64


class SpaceProjection(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(features_size, 512, dtype=float64),
                                   nn.ReLU(),
                                   nn.Linear(512, 256, dtype=float64),
                                   nn.ReLU(),
                                   nn.Linear(256, 512, dtype=float64),
                                   nn.ReLU(),
                                   nn.Linear(512, features_size, dtype=float64))

    def forward(self, x):
        out = self.model(x)
        return out
