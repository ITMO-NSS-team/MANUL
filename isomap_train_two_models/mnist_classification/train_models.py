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


def init_data(train_size=30000, validation_size=10000, test_size=10000):
    dataset = datasets.MNIST('../mnist_data', train=True, download=False)
    data = dataset.train_data.numpy() / 255
    labels = dataset.train_labels.numpy()
    data = np.expand_dims(data, axis=1)
    # CROP TRAIN SET
    train_labels = labels[:train_size]
    train_data = data[:train_size, :]
    # CROP VAL SET
    val_labels = labels[train_size:train_size+validation_size]
    val_data = data[train_size:train_size+validation_size, :]
    # CROP TEST SET
    test_labels = labels[:test_size]
    test_data = data[:test_size, :]
    # TRAIN LABELS TO PROBS
    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1
    # VAL LABELS TO PROBS
    val_labels_log = np.zeros((val_labels.shape[0], 10))
    for i in range(val_labels_log.shape[0]):
        val_labels_log[i][val_labels[i]] = 1
    # TEST LABELS TO PROBS
    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1
    return train_data, train_labels_log, val_data, val_labels_log, test_data, test_labels_log


train_features, train_target, val_features, val_target, test_features, test_target = init_data()

model_seq = ForecasterBase(input_size=(28, 28),
                           n_layers=3,
                           in_time_points=1,
                           out_time_points=1,
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

isomap_model = IsomapNN(dist_train)
isomap_model.to(device)

train_features = torch.tensor(train_features, dtype=float32).to(device)
train_target = torch.tensor(train_target, dtype=float32).to(device)
test_features = torch.tensor(test_features, dtype=float32).to(device)
test_target = torch.tensor(test_target, dtype=float32).to(device)

task_epochs = 2000

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

train_out = task_model(train_features)
train_out = train_out.reshape_as(train_target)
train_CEL = task_criterion(train_out, train_target)
print(f'Train cross entropy {train_CEL}')

train_out = torch.argmax(train_out, dim=1)
train_target = torch.argmax(train_target, dim=1)
traint_acc = accuracy(train_out, test_target)
print(f'Train accuracy {traint_acc}')


test_out = task_model(test_features)
test_out = test_out.reshape_as(test_target)
test_CEL = task_criterion(test_out, test_target)
print(f'Test cross entropy {test_CEL}')

test_out = torch.argmax(test_out, dim=1)
test_target = torch.argmax(test_target, dim=1)
test_acc = accuracy(test_out, test_target)
print(f'Test accuracy {test_acc}')


