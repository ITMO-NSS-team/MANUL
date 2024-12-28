import os.path
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn, float64
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from meta_manifold_learning.ProjectionModel import SpaceProjection
from regularizator.ModuleNN import ModelNN

device = 'cuda'

def generate_dataset():
    np.random.seed()
    # Step 1: Generate the dataset
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation,  y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
train_dataset = TensorDataset(torch.tensor(train_features, dtype=float64),
                              torch.tensor(train_target, dtype=float64))
train_loader = DataLoader(train_dataset, batch_size=300)

validation_dataset = TensorDataset(torch.tensor(validation_features, dtype=float64),
                              torch.tensor(validation_target, dtype=float64))
validation_loader = DataLoader(validation_dataset, batch_size=300)



folder = datetime.now().strftime('circless_class_%Y%m%d_%H.%M')
if not os.path.exists(folder):
    os.makedirs(folder)

features_size = train_features.shape[-1]
projection_model = SpaceProjection(features_size)
projection_model.to(device)
projection_optimizer = Adam(projection_model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
epochs = 1000
losses = []
val_losses = []

for epoch in range(epochs):
    epoch_losses = []
    for features, target in train_loader:
        features = features.to(device)
        target = target.to(device)
        projected_features = projection_model(features)
        projected_features = projected_features/torch.max(projected_features)

        task_model = ModelNN(train_feature=projected_features.cpu().detach().numpy(),
                             train_target=target,
                             problem='binary_class',
                             num_epochs=300,
                             stop_criteria_count=100)
        task_model.train()
        output = task_model.model(projected_features)
        loss = criterion(output, target.reshape_as(output))
        epoch_losses.append(loss.item())

        loss.backward()
        projection_optimizer.step()

    val_epoch_losses = []
    for val_features, val_target in validation_loader:
        val_features = val_features.to(device)
        val_target = val_target.to(device)
        projected_val_features = projection_model(val_features)
        projected_val_features = projected_val_features / torch.max(projected_val_features)

        task_model = ModelNN(train_feature=projected_val_features.cpu().detach().numpy(),
                             train_target=val_target,
                             problem='binary_class',
                             num_epochs=300,
                             stop_criteria_count=100)
        task_model.train()
        val_output = task_model.model(projected_val_features)
        val_loss = criterion(val_output, val_target.reshape_as(val_output))
        val_epoch_losses.append(val_loss.item())

    losses.append(np.mean(epoch_losses))
    val_losses.append(np.mean(val_epoch_losses))
    print(f'epoch {epoch}/{epochs}, loss={losses[-1]}, validation loss={val_losses[-1]}')

torch.save(projection_model.state_dict(), f'{folder}/reprojection_model.pt')
plt.plot(np.arange(epochs), losses, label='Train')
plt.plot(np.arange(epochs), val_losses, label='Validation')
plt.legend()
plt.show()

# RAW MODEL
task_model = ModelNN(train_feature=test_features,
                             train_target=test_target,
                             problem='binary_class',
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
fig.suptitle(f'Raw model: Train ROC AUC={train_acc}, Test ROC AUC={test_acc}')
plt.tight_layout()
plt.savefig(f'{folder}/raw_model_prediction.png')
plt.show()

# REPROJ MODEL
proj_train_features = projection_model(torch.tensor(train_features).to(device)).cpu().detach().numpy()
proj_test_features = projection_model(torch.tensor(test_features).to(device)).cpu().detach().numpy()
proj_train_features = proj_train_features/np.max(proj_train_features)
proj_test_features = proj_test_features/np.max(proj_test_features)

task_model = ModelNN(train_feature=proj_train_features,
                             train_target=train_target,
                             problem='binary_class',
                             num_epochs=300)

task_model.train()
train_acc = task_model.get_metric_on_train()
test_acc = task_model.get_metric_on_test(proj_test_features, test_target)
train_output = task_model.model(torch.tensor(proj_train_features).to('cuda')).cpu().detach().numpy()
output = task_model.model(torch.tensor(proj_test_features).to('cuda')).cpu().detach().numpy()
output = np.round(output)

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

fig.suptitle(f'NN transformed: Train ROC AUC={train_acc}, Test ROC AUC={test_acc}')
plt.tight_layout()
plt.savefig(f'{folder}/best_graph_prediction.png')
plt.show()

def to_polar(X):
  r=X[:,0]**2+X[:,1]**2
  phi=torch.arctan(X[:,1]/X[:,0])
  polar=torch.stack((r,phi),axis=1)
  return polar


x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)
grid = np.vstack([Y.ravel(), X.ravel()]).T

plt.scatter(grid[:, 1], grid[:, 0])
plt.title('Grid in euclidean coordinates')
plt.show()

polar_grid_features = to_polar(torch.tensor(grid))
plt.scatter(polar_grid_features[:, 1], polar_grid_features[:, 0])
plt.title('Grid in polar coordinates')
plt.show()

proj_grid_features = projection_model(torch.tensor(grid).to(device)).cpu().detach().numpy()
proj_grid_features = proj_grid_features/np.max(proj_grid_features)

plt.scatter(proj_grid_features[:, 1], proj_grid_features[:, 0])
plt.title('Grid in transformed coordinates')
plt.show()