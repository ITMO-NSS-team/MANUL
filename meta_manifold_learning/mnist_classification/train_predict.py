import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch import nn, optim, float64, float32
from torch.utils.data import TensorDataset, DataLoader
from torchcnnbuilder.models import ForecasterBase
from torchvision import datasets

device = 'cuda'


def calc_f1_score(prediction, target):
    f1 = f1_score(target, prediction, average='weighted')
    return f1


def get_metric(model, features, target):
    output = model(features).cpu().detach().numpy()[:, 0, :]
    output = np.argmax(output, axis=1)
    target = np.argmax(target, axis=1)
    metric = calc_f1_score(output, target)
    return metric


def init_data():
    dataset = datasets.MNIST('data', train=True, download=False)
    train_data = dataset.train_data.numpy()/255
    train_labels = dataset.train_labels.numpy()
    train_data = np.expand_dims(train_data, axis=1)

    train_labels = train_labels[:15000]
    train_data = train_data[:15000, :]

    test_data = dataset.test_data.numpy()/255
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


def train_task_model(train_data, train_labels_log, epochs=500):
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=float32), torch.tensor(train_labels_log, dtype=float32))
    train_loader = DataLoader(train_dataset, batch_size=300)

    task_model = ForecasterBase(input_size=(28, 28),
                                n_layers=3,
                                latent_shape=[1, 10],
                                in_time_points=1,
                                out_time_points=1,
                                latent_activation_function=nn.ReLU()).encoder
    print(task_model)
    task_model.to(device)
    epochs = epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(task_model.parameters(), lr=1e-5)

    losses = []

    for epoch in range(epochs):
        epoch_loss = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = task_model(features)
            loss = criterion(prediction.reshape_as(targets), targets)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        losses.append(np.mean(epoch_loss))

        print(f'epoch {epoch}/{epochs}, loss={losses[-1]}')


    plt.plot(np.arange(len(losses)), losses)
    #plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Convergence plot')
    plt.show()

    return task_model

'''train_data, train_labels, test_data, test_labels = init_data()
model = train_task_model(train_data, train_labels)
metric_on_train = get_metric(model, torch.tensor(train_data, dtype=float32).to(device), train_labels)
print(f'Train f1 = {metric_on_train}')
metric_on_test = get_metric(model, torch.tensor(test_data, dtype=float32).to(device), test_labels)
print(f'Test f1 = {metric_on_test}')'''