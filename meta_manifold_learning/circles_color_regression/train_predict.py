import os.path
from datetime import datetime
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from meta_manifold_learning.HybridTrainer import train_reprojector
from regularizator.ModuleNN import ModelNN

device = 'cuda'

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
folder = datetime.now().strftime('circless_regress_%Y%m%d_%H.%M')

if not os.path.exists(folder):
    os.makedirs(folder)

EPOCHS = 5

projection_model = train_reprojector(train_features,
                                     train_target,
                                     validation_features,
                                     validation_target,
                                     criterion=nn.L1Loss(),
                                     working_folder=f'{os.getcwd()}/{folder}',
                                     epochs=EPOCHS,
                                     task='regres')

task_model = ModelNN(train_feature=test_features,
                     train_target=test_target,
                     problem='regres',
                     num_epochs=300,
                     stop_criteria_count=100)

task_model.train()
train_acc = task_model.get_metric_on_train()
test_acc = task_model.get_metric_on_test(test_features, test_target)
output = task_model.model(torch.tensor(test_features).to('cuda'))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[0].set_title('Target classes')
axs[1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
axs[1].set_title('Predicted classes')
fig.suptitle(f'Raw model: Train MSE={train_acc}, Test MSE={test_acc}')
plt.tight_layout()
plt.savefig(f'{folder}/raw_model_prediction.png')
plt.show()


# REPROJ MODEL
proj_train_features = projection_model(torch.tensor(train_features).to(device)).cpu().detach().numpy()
proj_test_features = projection_model(torch.tensor(test_features).to(device)).cpu().detach().numpy()
proj_train_features = proj_train_features / np.max(proj_train_features)
proj_test_features = proj_test_features / np.max(proj_test_features)

task_model = ModelNN(train_feature=proj_train_features,
                     train_target=train_target,
                     problem='regres',
                     num_epochs=300)

task_model.train()
train_acc = task_model.get_metric_on_train()
test_acc = task_model.get_metric_on_test(proj_test_features, test_target)
train_output = task_model.model(torch.tensor(proj_train_features).to(device)).cpu().detach().numpy()
output = task_model.model(torch.tensor(proj_test_features).to(device)).cpu().detach().numpy()

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].scatter(proj_test_features[:, 1], proj_test_features[:, 0], c=test_target)
axs[0, 0].set_title('Reprojected target classes')
axs[0, 1].scatter(proj_test_features[:, 1], proj_test_features[:, 0], c=output)
axs[0, 1].set_title('Reprojected predicted classes')
axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[1, 0].set_title('Euclidean target classes')
axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output)
axs[1, 1].set_title('Euclidean predicted classes')

axs[2, 0].scatter(train_features[:, 1], train_features[:, 0], c=train_output)
axs[2, 0].set_title('Euclidean train classes')
axs[2, 1].scatter(proj_train_features[:, 1], proj_train_features[:, 0], c=train_output)
axs[2, 1].set_title('Reprojected train classes')

fig.suptitle(f'NN transformed: Train MSE={train_acc}, Test MSE={test_acc}')
plt.tight_layout()
plt.savefig(f'{folder}/best_graph_prediction.png')
plt.show()

def to_polar_correct(X):
    z = X[:, 0] + 1j*X[:, 1]
    phi = z.angle()
    r = z.abs()
    polar = torch.stack((r, phi), axis=1)
    return polar


def to_polar(X):
    z = X[:, 0] + 1j*X[:, 1]
    phi = z.angle() % (torch.pi/2)
    r = z.abs()
    polar = torch.stack((r, phi), axis=1)
    return polar

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
grid = np.vstack([Y.ravel(), X.ravel()]).T

plt.scatter(grid[:, 1], grid[:, 0])
for i in range(grid.shape[0]):
    plt.annotate(str(i), (grid[i, 1], grid[i, 0]))
plt.title('Grid in euclidean coordinates')
plt.savefig(f'{folder}/euq_grid.png')
plt.show()

polar_grid_features = to_polar(torch.tensor(grid))
plt.scatter(polar_grid_features[:, 1], polar_grid_features[:, 0])
for i in range(polar_grid_features.shape[0]):
    plt.annotate(str(i), (polar_grid_features[i, 1], polar_grid_features[i, 0]))
plt.title('Grid in polar coordinates')
plt.savefig(f'{folder}/polar_grid.png')
plt.show()

proj_grid_features = projection_model(torch.tensor(grid).to(device)).cpu().detach().numpy()
proj_grid_features = proj_grid_features / np.max(proj_grid_features)

plt.scatter(proj_grid_features[:, 1], proj_grid_features[:, 0])
for i in range(grid.shape[0]):
    plt.annotate(str(i), (proj_grid_features[i, 1], proj_grid_features[i, 0]))
plt.title('Grid in transformed coordinates')
plt.savefig(f'{folder}/proj_grid.png')
plt.show()