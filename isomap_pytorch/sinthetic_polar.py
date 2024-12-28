import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32
from torch.utils.data import TensorDataset


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

dict_train = pairwise_distances(train_features, train_features)
dist_train = torch.tensor(dict_train, dtype=float32)

polar_features = to_polar(torch.tensor(train_features))
polar_dict_train = pairwise_distances(polar_features, polar_features)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(dict_train)
axs[1].imshow(polar_dict_train)
plt.show()