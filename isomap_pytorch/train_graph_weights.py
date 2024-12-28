import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn, optim, float64
from torch.utils.data import TensorDataset, DataLoader

from isomap_pytorch.Isomap import IsomapNN
from regularizator.ModuleNN import ModelNN

device = 'cuda'
def to_polar(X):
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    phi = torch.arctan(X[:, 1] / X[:, 0])
    polar = torch.stack((r, phi), axis=1)
    return polar

def generate_dataset():
    np.random.seed()
    # Step 1: Generate the dataset
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
train_dataset = TensorDataset(torch.tensor(train_features, dtype=float32), torch.tensor(train_target, dtype=float32))
train_loader = DataLoader(train_dataset, batch_size=train_features.shape[0])

#dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)
polar_features = to_polar(torch.tensor(train_features))
dist_train = torch.tensor(pairwise_distances(polar_features, polar_features), dtype=float32)

isomap_model = IsomapNN(dist_train)

isomap_model.to(device)
isomap_criterion = nn.BCELoss()
isomap_optimizer = optim.AdamW(isomap_model.parameters(), lr=0.001)

working_folder = datetime.now().strftime('isomap_train_%Y%m%d_%H.%M')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)
working_folder = f'{os.getcwd()}/{working_folder}'

epochs = 500
save_each = 50
losses = []

best_lost = np.inf
best_model = None

for epoch in range(epochs):
    epoch_losses = []
    for features, target in train_loader:
        target = target.to(device)

        peproj_features = isomap_model()

        task_model = ModelNN(train_feature=peproj_features.cpu().detach().numpy(),
                             train_target=target,
                             problem='binary_class',
                             num_epochs=300,
                             stop_criteria_count=100)
        task_model.train()


        def closure():
            output = task_model.model(peproj_features)
            loss = isomap_criterion(output.to(torch.float32), target.reshape_as(output).to(torch.float32))
            epoch_losses.append(loss.item())
            loss.backward()
            return loss
        isomap_optimizer.step(closure)



    losses.append(np.mean(epoch_losses))
    print(f'epoch {epoch}/{epochs}, loss={losses[-1]}')
    if losses[-1] < best_lost:
        best_model = isomap_model
        best_lost = losses[-1]

    if epoch % save_each == 0:
        plt.plot(np.arange(len(losses)), losses, label='Train')
        plt.title('Convergence plot')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.axhline(best_lost, c='r', linestyle='dashed')
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{working_folder}/isomap_model_convergence.png')
        plt.show()


torch.save(best_model.state_dict(), f'{working_folder}/isomap_model.pt')

plt.plot(np.arange(len(losses)), losses, label='Train')
plt.title('Convergence plot')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.axhline(best_lost, c='r', linestyle='dashed')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'{working_folder}/isomap_model_convergence.png')
plt.show()


train_proj_points = best_model.predict(dist_train)
test_dist = torch.tensor(pairwise_distances(test_features, train_features))
test_proj_points = best_model.predict(test_dist)

task_model = ModelNN(train_feature=train_proj_points.cpu().detach().numpy(),
                             train_target=train_target,
                             problem='binary_class',
                             num_epochs=300,
                             stop_criteria_count=100)
task_model.train()
train_acc = task_model.get_metric_on_train()
test_acc = task_model.get_metric_on_test(test_proj_points.cpu().detach().numpy(), test_target)

train_output = task_model.model(train_proj_points.to(float64)).cpu().detach().numpy()
output = task_model.model(test_proj_points.to(float64)).cpu().detach().numpy()


test_proj_points = test_proj_points.cpu().detach().numpy()
train_proj_points = train_proj_points.cpu().detach().numpy()


fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].scatter(test_proj_points[:, 1], test_proj_points[:, 0], c=test_target)
axs[0, 0].set_title('Reprojected target classes')
axs[0, 1].scatter(test_proj_points[:, 1], test_proj_points[:, 0], c=output)
axs[0, 1].set_title('Reprojected predicted classes')
axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[1, 0].set_title('Euclidean target classes')
axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output)
axs[1, 1].set_title('Euclidean predicted classes')

axs[2, 0].scatter(train_features[:, 1], train_features[:, 0], c=train_output)
axs[2, 0].set_title('Euclidean train classes')
axs[2, 1].scatter(train_proj_points[:, 1], train_proj_points[:, 0], c=train_output)
axs[2, 1].set_title('Reprojected train classes')

fig.suptitle(f'NN transformed: Train ROC AUC={train_acc}, Test ROC AUC={test_acc}')
plt.tight_layout()
plt.savefig(f'{working_folder}/best_graph_prediction.png')
plt.show()

