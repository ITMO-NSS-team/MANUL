import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
from isomap_pytorch.HybridModel import Hybrid

device = 'cuda'
def to_polar(X):
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    phi = torch.arctan(X[:, 1] / X[:, 0])
    polar = torch.stack((r, phi), axis=1)
    return polar

def generate_dataset():
    n_samples = 1000
    xs = np.random.uniform(low=-1, high=1, size=n_samples)
    ys = np.random.uniform(low=-1, high=1, size=n_samples)
    points = np.vstack((xs, ys)).T

    colors = np.array([(abs(point[0])+abs(point[1]))/2 for point in points])

    '''plt.scatter(points[:, 1], points[:, 0], c=colors)
    plt.colorbar()
    plt.show()'''

    X=points
    y=colors

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
train_dataset = TensorDataset(torch.tensor(train_features, dtype=float32), torch.tensor(train_target, dtype=float32))

train_features = torch.tensor(train_features, dtype=float32)
train_target = torch.tensor(train_target, dtype=float32)

#train_loader = DataLoader(train_dataset, batch_size=train_features.shape[0])

dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)

model = Hybrid(dist_train, train_features.shape[-1])
model.to(device)


criterion = nn.L1Loss()
isomap_criterion = nn.L1Loss()

isomap_optimizer = optim.AdamW([model.isomap_params], lr=0.01)
task_optimizer = optim.AdamW(model.task_model.parameters(), lr=0.0001)

isomap_epochs = 1000
task_epochs = 100
save_each = 200

'''best_lost = np.inf
best_isomap_model = None'''

losses = []

for isomap_ep in range(isomap_epochs):
    task_optimizer.zero_grad()
    # features = train_features.to(device)
    target = train_target.to(device)

    points = model(isomap_step=True)
    with torch.no_grad():
        features = points.clone()

    task_losses = []
    for task_ep in range(task_epochs):
        task_optimizer.zero_grad()

        output = model(features, isomap_step=False)
        loss = criterion(output, target.reshape_as(output))

        task_losses.append(loss.item())
        loss.backward()
        task_optimizer.step()
        #print(task_ep)

    losses.append(np.mean(task_losses))

    output = model(features, isomap_step=False)
    isomap_loss = isomap_criterion(output, target.reshape_as(output))
    isomap_loss.backward()
    isomap_optimizer.step()
    print(f'Epoch {isomap_ep}/{isomap_epochs}, loss = {losses[-1]}')




