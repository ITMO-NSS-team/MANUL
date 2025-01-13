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

dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)

model = Hybrid(dist_train, train_features.shape[-1])

model.to(device)
criterion = nn.L1Loss()
isomap_criterion = nn.L1Loss()
lr = 0.001
isomap_optimizer = optim.AdamW([model.isomap_params], lr=lr)
task_optimizer = optim.AdamW(model.task_model.parameters(), lr=0.0001)

isomap_epochs = 5000
task_model_epochs = 150
save_each = 500


best_lost = np.inf
best_isomap_model = None


losses = []
for i_epoch in range(isomap_epochs):
    points = model(isomap_step=True)

    with torch.no_grad():
        features = points.clone()

    t_epoch_losses = []

    target = train_dataset.tensors[1].to(device)

    best_task_model = None
    best_task_loss = np.inf
    for t_epoch in tqdm(range(task_model_epochs)):
        output = model(features, isomap_step=False)
        loss = criterion(output.to(torch.float32), target.reshape_as(output).to(torch.float32))
        t_epoch_losses.append(loss.item())
        loss.backward()
        task_optimizer.step()

        if loss.item() < best_task_loss:
            best_task_model = model
            best_task_loss = loss.item()

    #print(f'Epoch {i_epoch}/{isomap_epochs}, best loss = {np.mean(t_epoch_losses)}')
    print(f'Epoch {i_epoch}/{isomap_epochs}, best loss = {best_task_loss}')
    losses.append(best_task_loss)


    '''# считаем еще один лосс потому что степ первого оптимизатора обнуляет градиент и второй не знает куда шагать
    output = model(features, isomap_step=False)
    isomap_loss = isomap_criterion(output.to(torch.float32), target.reshape_as(output).to(torch.float32))
    print(f'Epoch {i_epoch}/{isomap_epochs}, loss = {isomap_loss.item()}')
    isomap_loss.backward()'''
    isomap_optimizer.step()

    if losses[-1] < best_lost:
        best_isomap_model = best_task_model
        best_lost = losses[-1]

    if i_epoch % save_each == 0:
        plt.plot(np.arange(len(losses)), losses, label='Train')
        plt.title('Convergence plot')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.axhline(best_lost, c='r', linestyle='dashed')
        plt.annotate(str(round(best_lost, 6)), (0, best_lost), c='r')
        plt.legend()
        plt.tight_layout()
        #plt.savefig(f'{working_folder}/isomap_model_convergence.png')
        plt.show()


#torch.save(best_isomap_model.state_dict(), f'{working_folder}/isomap_model.pt')

plt.plot(np.arange(len(losses)), losses, label='Train')
plt.title('Convergence plot')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.axhline(best_lost, c='r', linestyle='dashed')
plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
plt.legend()
plt.tight_layout()
#plt.savefig(f'{working_folder}/isomap_model_convergence.png')
plt.show()

