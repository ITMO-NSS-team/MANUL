import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import float32, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchcnnbuilder.models import ForecasterBase

from meta_manifold_learning.mnist_classification.train_predict import train_task_model, init_data, get_metric

device = 'cuda'

train_data, train_labels, test_data, test_labels = init_data()

'''model = train_task_model(train_data, train_labels, epochs=1000)
metric_on_train = get_metric(model, torch.tensor(train_data, dtype=float32).to(device), train_labels)
print(f'Train f1 = {metric_on_train}')
metric_on_test = get_metric(model, torch.tensor(test_data, dtype=float32).to(device), test_labels)
print(f'Test f1 = {metric_on_test}')
'''
train_dataset = TensorDataset(torch.tensor(train_data, dtype=float32), torch.tensor(train_labels, dtype=float32))
train_loader = DataLoader(train_dataset, batch_size=5000)

epochs = 200
save_epoch_each = 10

working_folder = datetime.now().strftime('mnist_class_%Y%m%d_%H.%M')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)
working_folder = f'{os.getcwd()}/{working_folder}'


projection_model = ForecasterBase(input_size=(28, 28),
                                  n_layers=3,
                                  in_time_points=1,
                                  out_time_points=1,
                                  finish_activation_function=nn.ReLU())

projection_model.to(device)
criterion = nn.CrossEntropyLoss()
projection_optimizer = optim.AdamW(projection_model.parameters(), lr=1e-3)

losses = []

for epoch in tqdm(range(epochs)):
    epoch_losses = []
    for features, target in train_loader:
        features = features.to(device)
        target = target.to(device)
        projected_features = projection_model(features)
        projected_features = projected_features / torch.max(projected_features)

        task_model = train_task_model(projected_features, target, epochs=700)

        output = task_model(projected_features)
        loss = criterion(output.reshape_as(target), target)
        epoch_losses.append(loss.item())

        loss.backward()
        projection_optimizer.step()

    losses.append(np.mean(epoch_losses))
    print(f'epoch {epoch}/{epochs}, loss={losses[-1]}')

    if epoch % save_epoch_each == 0:
        if not os.path.exists(f'{working_folder}/models'):
            os.makedirs(f'{working_folder}/models')
        torch.save(projection_model.state_dict(), f'{working_folder}/models/reprojection_mode_ep{epoch}.pt')
        plt.plot(np.arange(len(losses)), losses, label='Train')
        plt.scatter(np.arange(0, len(losses), save_epoch_each), losses[::save_epoch_each], c='r')
        plt.title('Convergence plot reprojection model')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.tight_layout()
        plt.savefig(f'{working_folder}/reprojection_model_convergence.png')
        plt.show()

torch.save(projection_model.state_dict(), f'{working_folder}/reprojection_model.pt')
plt.plot(np.arange(len(losses)), losses, label='Train')
plt.scatter(np.arange(0, len(losses), save_epoch_each), losses[::save_epoch_each], c='r')
plt.title('Convergence plot')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(f'{working_folder}/reprojection_model_convergence.png')
plt.show()
