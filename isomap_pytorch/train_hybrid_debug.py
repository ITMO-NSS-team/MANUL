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

    colors = np.array([(abs(point[0]) + abs(point[1])) / 2 for point in points])

    '''plt.scatter(points[:, 1], points[:, 0], c=colors)
    plt.colorbar()
    plt.show()'''

    X = points
    y = colors

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
train_features = torch.tensor(train_features, dtype=float32)
train_target = torch.tensor(train_target, dtype=float32)

model = nn.Sequential(*[nn.Linear(train_features.shape[-1], 512, dtype=float32),
                        nn.Linear(512, 256, dtype=float32),
                        nn.Linear(256, 64, dtype=float32),
                        nn.Linear(64, 1, dtype=float32)])
model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs = 150

losses = []


best_task_model = None
best_task_loss = np.inf
for ep in range(epochs):
    optimizer.zero_grad()

    features = train_features.to(device)
    targets = train_target.to(device)

    output = model(features)
    loss = criterion(output.to(torch.float32), targets.reshape_as(output).to(torch.float32))
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    print(f'Epoch {ep}/{epochs}, loss = {losses[-1]}')

    if np.mean(losses[-1]) < best_task_loss:
        best_task_model = model
        best_task_loss = losses[-1]


plt.plot(np.arange(len(losses)), losses, label='Train')
plt.title('Convergence plot')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.axhline(best_task_loss, c='r', linestyle='dashed')
plt.annotate(str(round(best_task_loss, 4)), (0, best_task_loss), c='r')
plt.legend()
plt.tight_layout()
#plt.savefig(f'{working_folder}/isomap_model_convergence.png')
plt.show()
