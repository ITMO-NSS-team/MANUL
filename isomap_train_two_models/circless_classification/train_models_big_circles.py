import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn, optim



import time

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps


device = 'cuda'


def accuracy(predicted, target):
    predicted[predicted > 0.5] = 1
    predicted[predicted <= 0.5] = 0
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc

def plot_train_projection(train_points, reproj_points, predicted_classes, loss_value, filename):
    reproj_points = reproj_points.cpu().detach().numpy()
    try:
        train_points = train_points.cpu().detach().numpy()
        predicted_classes = predicted_classes.cpu().detach().numpy()
    except Exception as e:
        pass
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(train_points[:, 1], train_points[:, 0], c=predicted_classes)
    axs[0].set_title('Euclidean train classes')
    axs[1].scatter(reproj_points[:, 1], reproj_points[:, 0], c=predicted_classes)
    axs[1].set_title('Reprojected train classes')

    fig.suptitle(f'NN transformed: BCE={loss_value}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    #plt.show()


def to_polar(X):
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    phi = torch.arctan(X[:, 1] / X[:, 0])
    polar = torch.stack((r, phi), axis=1)
    return polar

def generate_dataset():
    np.random.seed()
    # Step 1: Generate the dataset
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


for _ in range(10):
    retain_points = 200

    train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
    train_features = torch.tensor(train_features, dtype=float32)
    train_target = torch.tensor(train_target, dtype=float32)
    test_target = torch.tensor(test_target, dtype=float32)
    validation_target = torch.tensor(validation_target, dtype=float32)

    model_seq = [nn.Linear(train_features.size(1), 512, dtype=float32),
                                     nn.Linear(512, 256, dtype=float32),
                                     nn.Linear(256, 64, dtype=float32),
                                     nn.Linear(64, 1, dtype=float32),
                                     nn.Sigmoid()]

    dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32).to(device)

    pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)

    reduced_train_target = train_target[pts]
    reduced_train_features = train_features[pts]

    test_dist = torch.tensor(pairwise_distances(test_features, reduced_train_features.cpu().detach().numpy()), dtype=float32).to(device)

    dist_train_new = torch.zeros((train_features.shape[0], train_features[pts].shape[0]),device=device)
    for i, x_train in enumerate(train_features):
        dist_train_new[i, :] = torch.linalg.norm(train_features[pts] - x_train, axis=1)

    isomap_model = IsomapNN(reduced_dist)
    isomap_model.to(device)

    isomap_criterion = nn.BCELoss()
    validation_criterion = nn.BCELoss()
    lr = 0.01
    isomap_optimizer = optim.AdamW(isomap_model.parameters(), lr=lr)

    working_folder = datetime.now().strftime('ICML_RESULTS/isomap_train_%Y%m%d_%H.%M')
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
        os.makedirs(f'{working_folder}/optimization')
    working_folder = f'{os.getcwd()}/{working_folder}'

    epochs = 500
    task_epochs = 50
    save_each = 100
    losses = []
    val_losses = []
    val_epochs = []

    best_lost = np.inf
    best_val_loss = np.inf
    best_isomap_model = None

    for epoch in range(epochs):
        epoch_losses = []
        target = train_target.to(device)
        reduced_target = reduced_train_target.to(device)

        reproj_features = isomap_model().to(float32)

        plt.scatter(reproj_features.cpu().detach().numpy()[:, 1],
                    reproj_features.cpu().detach().numpy()[:, 0],
                    c=reduced_train_target.cpu().detach().numpy())
        plt.show()

        with torch.no_grad():
            features = isomap_model.transform(dist_train_new)

        # ИНИЦИАЛИЗАЦИЯ ВТОРОЙ МОДЕЛИ
        task_model = nn.Sequential(*model_seq).to(device)
        task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
        task_criterion = nn.BCELoss()
        for ep in range(task_epochs):
            task_optim.zero_grad()
            out = task_model(features)
            task_loss = task_criterion(out, target.reshape_as(out))
            #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
            task_loss.backward()
            task_optim.step()
        output = task_model(reproj_features)
        loss = isomap_criterion(output.to(torch.float32), reduced_target.reshape_as(output).to(torch.float32))
        epoch_losses.append(loss.item())


        if loss > 0.15:
            for g in isomap_optimizer.param_groups:
                g['lr'] = 0.01
                lr = 0.01
        if 0.15 >= loss >= 0.01:
            for g in isomap_optimizer.param_groups:
                g['lr'] = 0.001
                lr = 0.001
        if loss <= 0.01:
            for g in isomap_optimizer.param_groups:
                g['lr'] = 0.0001
                lr = 0.0001

        losses.append(np.mean(epoch_losses))

        # VALIDATION
        reproj_features2 = isomap_model.transform(test_dist)
        test_output = task_model(reproj_features2)
        val_loss = validation_criterion(test_output.to(torch.float32).cpu(), test_target.reshape_as(test_output).to(torch.float32))
        val_losses.append(val_loss.item())
        val_epochs.append(epoch)

        # SAVE BEST MODEL
        if losses[-1] < best_lost:
            best_lost = losses[-1]
            plot_train_projection(reduced_train_features, reproj_features, output, losses[-1], f'{working_folder}/{epoch}_train.png')
            plot_train_projection(test_features, reproj_features2, test_target, val_losses[-1],
                                  f'{working_folder}/{epoch}_validation.png')

        if val_losses[-1] < best_val_loss:
            best_isomap_model = isomap_model
            best_val_loss = val_losses[-1]
            plot_train_projection(reduced_train_features, reproj_features, output, losses[-1],
                                  f'{working_folder}/{epoch}_train.png')
            plot_train_projection(test_features, reproj_features2, test_target, val_losses[-1],
                                  f'{working_folder}/{epoch}_validation.png')


        # SAVE OPTIMIZATION PROCESS
        np.save(f'{working_folder}/optimization/distance_matrix_{epoch}.npy', isomap_model.distances_matrix.cpu().detach().numpy())
        np.save(f'{working_folder}/optimization/train_reproj_points_{epoch}.npy', reproj_features.cpu().detach().numpy())


        loss.backward()
        isomap_optimizer.step()

        print(f'epoch {epoch}/{epochs}, lr={lr},  loss={losses[-1]}, val_loss={val_losses[-1]}')

        if epoch % save_each == 0:
            plt.figure()
            plt.plot(np.arange(len(losses)), losses, label='Train')
            plt.plot(val_epochs, val_losses, label='Validation')
            plt.title('Convergence plot')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.axhline(best_val_loss, c='green', linestyle='dashed')
            plt.axhline(best_lost, c='r', linestyle='dashed')
            plt.annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')
            plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
            plt.legend()
            plt.tight_layout()
            plt.yscale('log')
            plt.savefig(f'{working_folder}/isomap_model_convergence.png')
            plt.show()


    torch.save(best_isomap_model.state_dict(), f'{working_folder}/isomap_model.pt')

    plt.figure()
    plt.plot(np.arange(len(losses)), losses, label='Train')
    plt.plot(val_epochs, val_losses, label='Validation')
    plt.title('Convergence plot')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.axhline(best_val_loss, c='green', linestyle='dashed')
    plt.axhline(best_lost, c='r', linestyle='dashed')
    plt.annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')
    plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f'{working_folder}/isomap_model_convergence.png')
    plt.show()


    test_proj_points = best_isomap_model.transform(test_dist)
    train_reproj_points = best_isomap_model.transform(dist_train_new)

    test2_dist = torch.tensor(pairwise_distances(validation_features, reduced_train_features.cpu().detach().numpy()), dtype=float32).to(device)
    test2_proj_points = best_isomap_model.transform(test2_dist)

    # инициализация модели для теста
    task_model = nn.Sequential(*model_seq).to(device)
    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
    task_criterion = nn.BCELoss()
    for ep in range(task_epochs):
        task_optim.zero_grad()
        out = task_model(features)
        task_loss = task_criterion(out, target.reshape_as(out))
        #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
        task_loss.backward()
        task_optim.step()


    train_output = task_model(train_reproj_points.to(float32))
    output = task_model(test_proj_points.to(float32))
    test2_output = task_model(test2_proj_points.to(float32))

    test2_proj_points = test2_proj_points.cpu().detach().numpy()
    train_proj_points = train_reproj_points.cpu().detach().numpy()

    bce_score = nn.BCELoss()
    train_bce = bce_score(train_output, train_target.reshape_as(train_output).to(device)).item()
    test_bce = bce_score(output, test_target.reshape_as(output).to(device)).item()
    test2_bce = bce_score(test2_output, validation_target.reshape_as(output).to(device)).item()

    train_acc = accuracy(train_output.reshape_as(train_target).cpu(), train_target)
    test_acc = accuracy(output.reshape_as(test_target).cpu(), test_target)
    test2_acc = accuracy(test2_output.reshape_as(validation_target).cpu(), validation_target)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].scatter(test2_proj_points[:, 1], test2_proj_points[:, 0], c=validation_target)
    axs[0, 0].set_title('Test: Reprojected target classes')
    axs[0, 1].scatter(test2_proj_points[:, 1], test2_proj_points[:, 0], c=test2_output.cpu().detach().numpy())
    axs[0, 1].set_title('Test: Reprojected predicted classes')
    axs[1, 0].scatter(validation_features[:, 1], validation_features[:, 0], c=validation_target)
    axs[1, 0].set_title('Test: Euclidean target classes')
    axs[1, 1].scatter(validation_features[:, 1], validation_features[:, 0], c=test2_output.cpu().detach().numpy())
    axs[1, 1].set_title('Test: Euclidean predicted classes')

    axs[2, 0].scatter(train_features[:, 1], train_features[:, 0], c=train_output.cpu().detach().numpy())
    axs[2, 0].set_title('Train: Euclidean classes')
    axs[2, 1].scatter(train_proj_points[:, 1], train_proj_points[:, 0], c=train_output.cpu().detach().numpy())
    axs[2, 1].set_title('Train: Reprojected classes')

    fig.suptitle(f'NN transformed: Train BCE={train_bce}, Test BCE={test2_bce}\nTrain accuracy={train_acc}, Test accuracy={test2_acc}')
    plt.tight_layout()
    plt.savefig(f'{working_folder}/best_graph_prediction.png')
    plt.show()

    df = pd.DataFrame()
    df['BCE_train'] = [train_bce]
    df['BCE_validation'] = [test_bce]
    df['BCE_test'] = [test2_bce]
    df['acc_train'] = [train_acc.cpu().detach().numpy()]
    df['acc_validation'] = [test_acc.cpu().detach().numpy()]
    df['acc_test'] = [test2_acc.cpu().detach().numpy()]
    df.to_csv(f'{working_folder}/metrics.csv', index=False)

    def to_polar(X):
      r=X[:,0]**2+X[:,1]**2
      phi=torch.arctan(X[:,1]/X[:,0])
      polar=torch.stack((r,phi),axis=1)
      return polar


    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([Y.ravel(), X.ravel()]).T

    plt.scatter(grid[:, 1], grid[:, 0])
    plt.title('Grid in euclidean coordinates')
    plt.show()

    polar_grid_features = to_polar(torch.tensor(grid))
    plt.scatter(polar_grid_features[:, 1], polar_grid_features[:, 0])
    plt.title('Grid in polar coordinates')
    plt.show()

    grid_dist = pairwise_distances(grid, train_features[pts])

    proj_grid_features = best_isomap_model.transform(torch.tensor(grid_dist, dtype=float32).to(device)).cpu().detach().numpy()

    plt.scatter(proj_grid_features[:, 1], proj_grid_features[:, 0])
    plt.title('Grid in transformed coordinates')
    plt.savefig(f'{working_folder}/grid_with_isomap.png')
    plt.show()

