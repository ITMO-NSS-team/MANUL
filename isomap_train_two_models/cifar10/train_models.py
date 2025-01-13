import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchcnnbuilder.models import ForecasterBase
from torchvision import datasets

from isomap_train_two_models.Isomap import IsomapNN

device = 'cuda'


def accuracy(predicted, target):
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc


def init_data():
    dataset = datasets.CIFAR10('data', train=True, download=False)
    data = dataset.data / 255
    labels = np.array(dataset.targets)

    train_labels = labels[:20000]
    train_data = data[:20000, :]

    test_labels = labels[-10000:]
    test_data = data[-10000:]

    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1


    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1

    train_data = np.swapaxes(np.swapaxes(train_data, 3, 1), 3, 2)
    test_data = np.swapaxes(np.swapaxes(test_data, 3, 1), 3, 2)

    return train_data, train_labels_log, test_data, test_labels_log


train_features, train_target, test_features, test_target = init_data()

model_seq = ForecasterBase(input_size=(32, 32),
                           n_layers=3,
                           in_time_points=3,
                           out_time_points=3,
                           latent_shape=(1, 10)).encoder
print(model_seq)

rav_train_features = train_features.reshape(train_features.shape[0],
                                            train_features.shape[1] *
                                            train_features.shape[2] *
                                            train_features.shape[3])
rav_train_target = test_features.reshape(test_features.shape[0],
                                            test_features.shape[1] *
                                            test_features.shape[2] *
                                            test_features.shape[3])
dist_train = torch.tensor(pairwise_distances(rav_train_features, rav_train_features), dtype=float32)
test_dist = torch.tensor(pairwise_distances(rav_train_target, rav_train_features), dtype=float32).to(device)

rav_train_features = None
rav_train_target = None

#isomap_model = IsomapNN(dist_train)
#isomap_model.to(device)

train_features = torch.tensor(train_features, dtype=float32).to(device)
train_target = torch.tensor(train_target, dtype=float32).to(device)
test_features = torch.tensor(test_features, dtype=float32).to(device)
test_target = torch.tensor(test_target, dtype=float32).to(device)

task_epochs = 10000

task_model = nn.Sequential(*model_seq).to(device)
task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
task_criterion = nn.CrossEntropyLoss()

task_losses = []

for ep in range(task_epochs):
    task_optim.zero_grad()
    out = task_model(train_features)
    task_loss = task_criterion(out.reshape_as(train_target), train_target)
    print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
    task_loss.backward()
    task_optim.step()
    task_losses.append(task_loss.item())

plt.plot(np.arange(len(task_losses)), task_losses)
plt.show()


test_out = task_model(test_features)
test_out = test_out.reshape_as(test_target)
test_CEL = task_criterion(test_out, test_target)
print(f'Test cross entropy {test_CEL}')

test_out = torch.argmax(test_out, dim=1)
test_target = torch.argmax(test_target, dim=1)

test_acc = accuracy(test_out, test_target)
print(f'Test accuracy {test_acc}')


