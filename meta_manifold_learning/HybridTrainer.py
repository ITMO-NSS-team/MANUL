import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, float64
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from meta_manifold_learning.ProjectionModel import SpaceProjection
from regularizator.ModuleNN import ModelNN


def train_reprojector(train_features: np.ndarray,
                      train_target: np.ndarray,
                      validation_features: np.ndarray,
                      validation_target: np.ndarray,
                      criterion: nn.Module,
                      working_folder: str,
                      task: str = 'binary_class',
                      save_epoch_each: int = 10,
                      epochs: int = 500):
    device = 'cuda'
    os.makedirs(f'{working_folder}/models')
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=float64),
                                  torch.tensor(train_target, dtype=float64))
    train_loader = DataLoader(train_dataset, batch_size=300)
    validation_dataset = TensorDataset(torch.tensor(validation_features, dtype=float64),
                                       torch.tensor(validation_target, dtype=float64))
    validation_loader = DataLoader(validation_dataset, batch_size=300)

    features_size = train_features.shape[-1]
    projection_model = SpaceProjection(features_size)
    projection_model.to(device)
    projection_optimizer = Adam(projection_model.parameters(), lr=1e-3)
    criterion = criterion

    losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_losses = []
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            projected_features = projection_model(features)
            projected_features = projected_features / torch.max(projected_features)

            task_model = ModelNN(train_feature=projected_features.cpu().detach().numpy(),
                                 train_target=target,
                                 problem='binary_class',
                                 num_epochs=300,
                                 stop_criteria_count=100)
            task_model.train()
            output = task_model.model(projected_features)
            loss = criterion(output, target.reshape_as(output))
            epoch_losses.append(loss.item())

            loss.backward()
            projection_optimizer.step()

        val_epoch_losses = []
        for val_features, val_target in validation_loader:
            val_features = val_features.to(device)
            val_target = val_target.to(device)
            projected_val_features = projection_model(val_features)
            projected_val_features = projected_val_features / torch.max(projected_val_features)

            task_model = ModelNN(train_feature=projected_val_features.cpu().detach().numpy(),
                                 train_target=val_target,
                                 problem=task,
                                 num_epochs=300,
                                 stop_criteria_count=100)
            task_model.train()
            val_output = task_model.model(projected_val_features)
            val_loss = criterion(val_output, val_target.reshape_as(val_output))
            val_epoch_losses.append(val_loss.item())

        losses.append(np.mean(epoch_losses))
        val_losses.append(np.mean(val_epoch_losses))
        print(f'epoch {epoch}/{epochs}, loss={losses[-1]}, validation loss={val_losses[-1]}')

        if epoch % save_epoch_each == 0:
            torch.save(projection_model.state_dict(), f'{working_folder}/models/reprojection_mode_ep{epoch}.pt')

    torch.save(projection_model.state_dict(), f'{working_folder}/reprojection_model.pt')
    plt.plot(np.arange(epochs), losses, label='Train')
    plt.plot(np.arange(epochs), val_losses, label='Validation')
    plt.scatter(np.arange(0, epochs, save_epoch_each), losses[::10], c='r')
    plt.title('Convergence plot')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{working_folder}/reprojection_model_convergence.png')
    plt.show()

    return projection_model
