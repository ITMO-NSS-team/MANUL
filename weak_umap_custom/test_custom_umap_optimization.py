import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from isomap_train_two_models.synthetic_geometries.data_generation import torus, circles_2d
from weak_umap_custom.CustomUmap import CustomUmap

def to_polar(X):
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    phi = torch.arctan(X[:, 1] / X[:, 0])
    polar = torch.stack((r, phi), axis=1)
    return polar


data, colors = circles_2d()
data = torch.tensor(data)

polar_data = to_polar(data)
target_dist = torch.cdist(polar_data, polar_data)
target_dist = torch.tensor(target_dist, requires_grad=True)

model = CustomUmap(data)
optim = torch.optim.AdamW(params=model.parameters(), lr=0.001)

criterion = nn.MSELoss()

losses = []

epochs = 100
for ep in range(epochs):
    pred = model()
    pred_dist = torch.cdist(pred, pred)
    loss = criterion(pred_dist, target_dist)
    losses.append(loss.item())
    print(f'Epoch {ep}/{epochs} loss={loss.item()}')
    loss.backward()
    optim.step()
    optim

plt.plot(np.arange(len(losses)), losses)
plt.show()
