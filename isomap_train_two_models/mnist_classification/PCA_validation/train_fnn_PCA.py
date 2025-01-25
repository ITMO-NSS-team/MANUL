import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchvision import datasets

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps

from sklearn.decomposition import PCA

device = 'cuda'


def accuracy(predicted, target):
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc


def init_data(train_size=20000, test_size=10000):
    dataset = datasets.MNIST('../data', train=True, download=False)
    #dataset = datasets.FashionMNIST('data_fashion', train=True, download=True)
    train_data = dataset.train_data.numpy() / 255
    train_labels = dataset.train_labels.numpy()
    train_data = np.expand_dims(train_data, axis=1)
    # CROP TRAIN SET
    train_labels = train_labels[:train_size]
    train_data = train_data[:train_size, :]
    # INIT TEST
    test_data = dataset.test_data.numpy() / 255
    test_labels = dataset.test_labels.numpy()
    test_data = np.expand_dims(test_data, axis=1)
    # TRAIN LABELS TO PROBS
    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1
    # CROP TEST SET
    test_labels = test_labels[:test_size]
    test_data = test_data[:test_size, :]
    # TEST LABELS TO PROBS
    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1
    return train_data, train_labels_log, test_data, test_labels_log


train_features, train_target, test_features, test_target = init_data()
train_features = train_features.reshape(train_features.shape[0],
                                            train_features.shape[1] *
                                            train_features.shape[2] *
                                            train_features.shape[3])
test_features = test_features.reshape(test_features.shape[0],
                                            test_features.shape[1] *
                                            test_features.shape[2] *
                                            test_features.shape[3])
#retain_points = 1000

latent_len = 200

train_points_2d = torch.tensor(PCA(n_components=latent_len).fit_transform(train_features), dtype=float32).to(device)

model_seq = [nn.Linear(latent_len, 512, dtype=float32),
                                 nn.Linear(512, 256, dtype=float32),
                                 nn.Linear(256, 64, dtype=float32),
                                 nn.Linear(64, 10, dtype=float32),  # 10 classes
                                 ]
print(model_seq)

task_model = nn.Sequential(*model_seq).to(device)
task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
task_criterion = nn.CrossEntropyLoss()

train_features = torch.tensor(train_features, dtype=float32).to(device)
train_target = torch.tensor(train_target, dtype=float32).to(device)

test_target = torch.tensor(test_target, dtype=float32).to(device)

task_epochs = 300

task_losses = []
for ep in range(task_epochs):
    task_optim.zero_grad()
    out = task_model(train_points_2d)
    task_loss = task_criterion(out.reshape_as(train_target), train_target)
    print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
    task_loss.backward()
    task_optim.step()
    task_losses.append(task_loss.item())

plt.plot(np.arange(len(task_losses)), task_losses)
plt.show()

print(f'Train cross entropy {task_losses[-1]}')
out = torch.argmax(out, dim=1)
train_target = torch.argmax(train_target, dim=1)

train_acc = accuracy(out, train_target)
print(f'Test accuracy {train_acc}')

proj_test = PCA(n_components=latent_len).fit_transform(test_features)
proj_test = torch.tensor(proj_test, dtype=float32).to(device)
test_out = task_model(proj_test)
test_out = test_out.reshape_as(test_target)
test_CEL = task_criterion(test_out, test_target).item()
print(f'Test cross entropy {test_CEL}')

test_out = torch.argmax(test_out, dim=1)
test_target = torch.argmax(test_target, dim=1)

test_acc = accuracy(test_out, test_target)
print(f'Test accuracy {test_acc}')

try:
    df = pd.read_csv('PCA_convergence.csv')
except Exception:
    df = pd.DataFrame()
df[latent_len] = task_losses
df.to_csv('PCA_validation/PCA_convergence.csv', index=False)

try:
    df = pd.read_csv('PCA_N_comp_metrics.csv')
except Exception:
    df = pd.DataFrame()
df[latent_len] = [task_losses[-1], float(train_acc), float(test_CEL), float(test_acc)]
df['metric'] = ['train_CEL', 'train_acc', 'test_CEL', 'test_acc']
df.to_csv('PCA_validation/PCA_N_comp_metrics.csv', index=False)