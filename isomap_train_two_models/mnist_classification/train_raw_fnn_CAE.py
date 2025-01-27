import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchvision import datasets

#from isomap_train_two_models.Isomap import IsomapNN
#from isomap_train_two_models.utils import reduce_dist_fps

from sklearn.decomposition import PCA

device = 'cuda'

def plot_train_projection(train_points, predicted_classes, real_classes, loss_value, acc_value, filename):
    train_points = train_points.cpu().detach().numpy()

    train_points_2d = PCA(n_components=2).fit_transform(train_points)

    predicted_classes = predicted_classes.cpu().detach().numpy()
    real_classes = real_classes.cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=real_classes)
    axs[0].set_title('Euclidean (target)')
    axs[1].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=predicted_classes)
    axs[1].set_title('Euclidean (predicted)')

    fig.suptitle(f'MNIST (raw model)\n CrossEntropyLoss={round(loss_value, 4)}, '
                 f'accuracy={np.round(acc_value.item(), 4)}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


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


metrics = {'CEL_train': [],
           'CEL_test': [],
           'acc_train': [],
           'acc_test': []}



# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent





for i in range(5):
    train_features, train_target, _, _, test_features, test_target = init_data()

    latent_len = 8

    cae_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_features), torch.Tensor(train_target))
    cae_loader = torch.utils.data.DataLoader(cae_dataset, batch_size=128, shuffle=True)

    cae_model = ConvAutoencoder(latent_dim=latent_len).to("cuda")

    cae_criterion = nn.MSELoss()
    cae_optimizer = torch.optim.Adam(cae_model.parameters(), lr=1e-3)


    #Training loop
    epochs = 250
    for epoch in range(epochs):
    #ep=0
    #total_loss=len(cae_loader)
    #while (total_loss/len(cae_loader))>1e-3:
        total_loss = 0
        for batch, _ in cae_loader:
            batch = batch.to("cuda")
            cae_optimizer.zero_grad()
            reconstruction, _ = cae_model(batch)
            loss = cae_criterion(reconstruction, batch)
            loss.backward()
            cae_optimizer.step()
            total_loss += loss.item()
        #ep+=1
        print(f"CAE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(cae_loader)}")
        #print(f"CAE Epoch {ep}, Loss: {total_loss/len(cae_loader)}")

    with torch.no_grad():
        _,train_features = cae_model(torch.Tensor(train_features).to('cuda'))
        _,test_features = cae_model(torch.Tensor(test_features).to('cuda'))



    model_seq = [nn.Linear(train_features.shape[-1], 512, dtype=float32),
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
    test_features = torch.tensor(test_features, dtype=float32).to(device)
    test_target = torch.tensor(test_target, dtype=float32).to(device)

    task_epochs = 300

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

    print(f'Train cross entropy {task_losses[-1]}')
    out = torch.argmax(out, dim=1)
    train_target = torch.argmax(train_target, dim=1)
    train_acc = accuracy(out, train_target)
    print(f'Train accuracy {train_acc}')

    test_out = task_model(test_features)
    test_out = test_out.reshape_as(test_target)
    test_CEL = task_criterion(test_out, test_target).item()
    print(f'Test cross entropy {test_CEL}')

    test_out = torch.argmax(test_out, dim=1)
    test_target = torch.argmax(test_target, dim=1)
    test_acc = accuracy(test_out, test_target)
    print(f'Test accuracy {test_acc}')

    metrics['CEL_train'].append(task_losses[-1])
    metrics['CEL_test'].append(test_CEL)
    metrics['acc_train'].append(train_acc.cpu().detach().numpy())
    metrics['acc_test'].append(test_acc.cpu().detach().numpy())

    working_folder = datetime.now().strftime('ICML_RESULTS/mnist_fnn_(raw_model)%Y%m%d_%H.%M')
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    plot_train_projection(train_features, out, train_target, task_losses[-1], train_acc, f'{working_folder}/train_{i}.png')
    plot_train_projection(test_features, test_out, test_target, test_CEL, test_acc, f'{working_folder}/test_{i}.png')

df = pd.DataFrame()
df['CEL_train'] = metrics['CEL_train']
df['CEL_test'] = metrics['CEL_test']
df['acc_train'] = metrics['acc_train']
df['acc_test'] = metrics['acc_test']
df.to_csv(f'ICML_RESULTS/metrics_raw_model.csv', index=False)