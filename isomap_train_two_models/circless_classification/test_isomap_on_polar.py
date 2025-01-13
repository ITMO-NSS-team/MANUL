import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn, optim, float64
from torch.utils.data import TensorDataset, DataLoader

from isomap_train_two_models.Isomap import IsomapNN

device = 'cuda'

def plot_train_projection(train_points, reproj_points, predicted_classes, loss_value, filename):
    try:
        train_points = train_points.cpu().detach().numpy()
        predicted_classes = predicted_classes.cpu().detach().numpy()
    except Exception as e:
        pass
    reproj_points = reproj_points.cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(train_points[:, 1], train_points[:, 0], c=predicted_classes)
    axs[0].set_title('Euclidean  classes')
    axs[1].scatter(reproj_points[:, 1], reproj_points[:, 0], c=predicted_classes)
    axs[1].set_title('Reprojected  classes')

    #fig.suptitle(f'NN transformed: Train BCE={loss_value}')
    plt.tight_layout()
    #plt.savefig(filename)
    plt.show()


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


train_features = torch.tensor(train_features, dtype=float32)
train_features_polar = to_polar(train_features)

train_target = torch.tensor(train_target, dtype=float32)

dist_train = torch.tensor(pairwise_distances(train_features, train_features_polar), dtype=float32).to('cuda')

isomap_model = IsomapNN(dist_train)


reproj_features = isomap_model()
plot_train_projection(train_features, reproj_features, train_target, '', '')


new_dist = torch.tensor(pairwise_distances(train_features.cpu().detach().numpy(), train_features_polar.cpu().detach().numpy()), dtype=float32).to('cuda')
reproj_features2 = isomap_model.transform(new_dist)
plot_train_projection(train_features, reproj_features2, train_target, '', '')


test_dist = torch.tensor(pairwise_distances(test_features, train_features.cpu().detach().numpy()), dtype=float32).to('cuda')
'''test_dist = torch.zeros((test_features.shape[0], train_features.shape[0]))  # Create an empty matrix for test-to-train distances
for i, x_test in enumerate(test_features):
    test_dist[i, :] = torch.linalg.norm(train_features - x_test, axis=1)'''

reproj_features3 = isomap_model.transform(test_dist)
plot_train_projection(test_features, reproj_features3, test_target, '', '')