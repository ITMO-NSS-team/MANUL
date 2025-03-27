import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchcnnbuilder.models import ForecasterBase

device = 'cuda'

data_path = 'C:/Users/Julia/Documents/NSS_lab/TorchCNNBuilder/examples/data/MovingMNIST/mnist_test_seq.npy'
ds = np.load(data_path)[:, :5000, :, :]/255

ds = resize(ds, (ds.shape[0], ds.shape[1], ds.shape[2]//2, ds.shape[3]//2))

cae = ForecasterBase(input_size=(ds.shape[2], ds.shape[3]),
                     n_layers=3,
                     in_time_points=1,
                     out_time_points=1,
                     latent_shape=(64, 1),
                     finish_activation_function=nn.ReLU()).to(device)
print(cae)
ds = np.reshape(ds, (ds.shape[0]*ds.shape[1], ds.shape[2], ds.shape[3]))
ds = np.expand_dims(ds, axis=1)
ds = TensorDataset(torch.tensor(ds, dtype=torch.float32))
train_loader = DataLoader(ds, batch_size=300, shuffle=False)
optim = torch.optim.Adam(params=cae.parameters(), lr=0.00001)
criterion = nn.MSELoss()

losses = []
best_model = None
best_loss = np.inf
epochs = 10000
for ep in range(epochs):
    batch_loss = []
    for batch in train_loader:
        batch = batch[0].to(device)
        pred = cae(batch)
        loss = criterion(pred, batch)
        batch_loss.append(loss.item())
        loss.backward()
        optim.step()
    losses.append(np.mean(batch_loss))
    print(f'Epoch {ep}/{epochs}, loss = {losses[-1]}')
    if losses[-1] < best_loss:
        best_loss = losses[-1]
        best_model = cae
        torch.save(cae.state_dict(), 'cae_1_best_model(latent64).pt')



    if ep % 500 == 0:
        plt.imshow(pred[0][0].cpu().detach().numpy())
        plt.colorbar()
        plt.show()
        plt.imshow(batch[0][0].cpu().detach().numpy())
        plt.colorbar()
        plt.show()

        plt.plot(np.arange(len(losses)), losses)
        plt.title('Convergence CAE moving_mnist')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
