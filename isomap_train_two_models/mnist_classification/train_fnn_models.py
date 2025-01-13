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
    dataset = datasets.MNIST('data', train=True, download=False)
    #dataset = datasets.FashionMNIST('data_fashion', train=True, download=True)
    train_data = dataset.train_data.numpy() / 255
    train_labels = dataset.train_labels.numpy()
    train_data = np.expand_dims(train_data, axis=1)

    train_labels = train_labels[:2000]
    train_data = train_data[:2000, :]

    test_data = dataset.test_data.numpy() / 255
    test_labels = dataset.test_labels.numpy()

    test_data = np.expand_dims(test_data, axis=1)

    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1

    test_labels = test_labels[:3000]
    test_data = test_data[:3000, :]

    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1

    return train_data, train_labels_log, test_data, test_labels_log


train_features, train_target, test_features, test_target = init_data()



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

latent_len = 10
torch.cuda.memory_summary()
#orch.cuda.empty_cache()
dist_train=dist_train.to('cpu')
isomap_model = IsomapNN(dist_train, n_components=latent_len)
isomap_model.to(device)

model_seq = [nn.Linear(latent_len, 512, dtype=float32),
                                 nn.Linear(512, 256, dtype=float32),
                                 nn.Linear(256, 64, dtype=float32),
                                 nn.Linear(64, 10, dtype=float32),  # 10 classes
                                 ]
print(model_seq)

#train_features = torch.tensor(train_features, dtype=float32).to(device)
train_target = torch.tensor(train_target, dtype=float32).to(device)
#test_features = torch.tensor(test_features, dtype=float32).to(device)
test_target = torch.tensor(test_target, dtype=float32).to(device)

isomap_epochs = 100
task_epochs = 200

lr = 0.01
isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=lr)
isomap_criterion = nn.CrossEntropyLoss()
losses = []

for epoch in range(isomap_epochs):

    reproj_features = isomap_model().to(float32)
    with torch.no_grad():
        features = reproj_features.clone()

    task_model = nn.Sequential(*model_seq).to(device)
    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
    task_criterion = nn.CrossEntropyLoss()

    task_losses = []
    for ep in range(task_epochs):
        task_optim.zero_grad()
        out = task_model(features)
        task_loss = task_criterion(out.reshape_as(train_target), train_target)
        print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
        task_loss.backward()
        task_optim.step()
        task_losses.append(task_loss.item())

    #plt.plot(np.arange(len(task_losses)), task_losses)
    #plt.show()

    output = task_model(reproj_features)
    isomap_loss = isomap_criterion(output.to(torch.float32), train_target.reshape_as(output).to(torch.float32))
    losses.append(isomap_loss.item())

    isomap_loss.backward()
    isomap_optim.step()

    print(f'epoch {epoch}/{isomap_epochs}, lr={lr},  loss={losses[-1]}')

'''test_out = task_model(test_features)
test_out = test_out.reshape_as(test_target)
test_CEL = task_criterion(test_out, test_target)
print(f'Test cross entropy {test_CEL}')

test_out = torch.argmax(test_out, dim=1)
test_target = torch.argmax(test_target, dim=1)

test_acc = accuracy(test_out, test_target)
print(f'Test accuracy {test_acc}')'''


