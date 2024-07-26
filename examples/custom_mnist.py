from datetime import datetime
from typing import Callable
from tqdm import tqdm

import numpy as np
from SALib import ProblemSpec
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, f1_score
import torch.nn as nn
from torch import randperm, tensor
from torch.optim import Adam
from torch import float64 as fl64

from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def get_data():
    features = np.load("data/feature_mnist.npy")
    target = np.load("data/target_mnist.npy")
    angles = np.load("data/angle_mnist.npy")
    # data is already shuffled for class balance
    new_features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))
    new_feature = []
    new_target = []
    new_angles = []
    for i, elem in enumerate(target):
        if elem not in [9]:
            # remove 9 from classification  because augmentation makes it equal to 6
            new_feature.append(new_features[i])
            new_target.append(elem)
            new_angles.append(angles[i])
    samples_num = 20000
    new_feature = np.array(new_feature[:samples_num], dtype='int64')
    new_target = np.array(new_target[:samples_num])
    new_angles = np.array(new_angles[:samples_num])
    return new_feature, new_target, new_angles


def get_nn_model(input_dim):
    model = nn.Sequential(nn.Linear(input_dim, 512, dtype=fl64),
                          nn.ReLU(),
                          nn.Linear(512, 128, dtype=fl64),
                          nn.ReLU(),
                          nn.Dropout(p=0.25),
                          nn.Linear(128, 10, dtype=fl64),
                          nn.Softmax(dim=1))
    return model


def get_cnn_model():
    model = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, dtype=fl64),
                          nn.ReLU(),
                          nn.Conv2d(32, 1, kernel_size=5, dtype=fl64),
                          nn.Flatten(),
                          nn.Linear(400, 10, dtype=fl64),
                          nn.Softmax(dim=1))
    return model


def f1_loss(target, model_output):
    # output = model_output
    # max_possible_labels = np.argmax(output, axis=1)
    # output = max_possible_labels
    return f1_score(target, model_output, average='weighted')


def split_dataset(data, split_ratio=0.8):
    split_ratio = int(data.shape[0] * split_ratio)
    train = data[:split_ratio]
    test = data[split_ratio:]
    return train, test


feature, target, angles = get_data()
feature = feature.reshape(feature.shape[0], int(feature.shape[1] ** 0.5), int(feature.shape[1] ** 0.5))
feature = feature[:, None, :, :]
train_features, test_features = split_dataset(feature)
train_target, test_target = split_dataset(target)

model_structure = get_cnn_model()

model = ModelNN(model_structure=model_structure,
                train_feature=train_features.astype(float),
                train_target=train_target.astype(float),
                problem="multiclass",
                target_metric=f1_loss)
model.train(num_epochs=30, plot_convergence=True)
print(model.get_metric_on_train())
print(model.get_metric_on_test(test_features.astype(float), test_target.astype(float)))

base_individ = DataStructureGraph(data=train_features.reshape(train_features.shape[0], 28 * 28),
                                  cash_folder='mnist_custom_nn',
                                  n_neighbors=20,
                                  )
#base_individ.show_2d(train_target)
model.features = train_features[base_individ.basis].astype(float)
model.target = train_target[base_individ.basis].astype(float)
model.train(num_epochs=200, graph=base_individ, plot_convergence=True)
print(model.get_metric_on_train())
print(model.get_metric_on_test(test_features, test_target))
