
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pmlb import fetch_data
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps


def plot_predictoion_PCA_transform(points, proj_points, true_labels, predicted_labels, title, save_path):
    points_2d = PCA(n_components=2).fit_transform(points)
    proj_points_2d = PCA(n_components=2).fit_transform(proj_points)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    cs0 = axs[0].scatter(points_2d[:, 1], points_2d[:, 0], c=true_labels)
    fig.colorbar(cs0, ax=axs[0])
    axs[0].set_title('Euclidean - Target values')

    cs1 = axs[1].scatter(proj_points_2d[:, 1], proj_points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs1, ax=axs[1])
    axs[1].set_title('ISOMAP projected - Predicted values')

    cs2 = axs[2].scatter(points_2d[:, 1], points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs2, ax=axs[2])
    axs[2].set_title('Euclidean - Predicted values')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_data(df_name, save_path=None):
    """
    normalize data per column
    """
    if df_name == '0_MY_CIRCLES':
        df = pd.read_csv(f'{save_path}/{df_name}.csv')
    else:
        df = fetch_data(df_name)
    df = df.drop_duplicates()
    if save_path is not None:
        df.to_csv(f'{save_path}/{df_name}.csv', index=False)
    for column in df.columns:
        df[column] = df[column] / df[column].max()
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(f'{save_path}/{df_name}_normalized.csv', index=False)

    y = df['target'].to_numpy()
    X = df[df.columns.drop('target')].to_numpy()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if df_name == '0_MY_CIRCLES':
        return X_train, X_test, np.expand_dims(y_train, axis=1), np.expand_dims(y_test, axis=1)
    else:
        return X_train, X_test, y_train, y_test


def run_isomap(run_number=5):
    device = 'cuda'
    # ds_names = call_df_names('regression')
    for ds_name in os.listdir('regression'):
        ds_path = f'regression/{ds_name}/isomap_no_optimization'
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        X_train, X_test, y_train, y_test = get_data(ds_name,save_path=f'regression/{ds_name}')

        init_assump = torch.tensor(pairwise_distances(np.expand_dims(y_train, axis=1), np.expand_dims(y_train, axis=1), metric='l1'), dtype=float32)

        # SELECT SPARSE POINTS
        if X_train.shape[0] > 1000:
            retain_points = 1000
            pts, reduced_dist = reduce_dist_fps(init_assump, retain_points)
        else:
            reduced_dist = torch.tensor(init_assump, dtype=float32).to(device)
            pts = np.arange(X_train.shape[0])

        dist_train_new = torch.tensor(pairwise_distances(X_train, X_train[pts]), dtype=float32).to(device)
        test_dist = torch.tensor(
            pairwise_distances(X_test, X_train[pts]), dtype=float32).to(device)

        train_target = torch.tensor(y_train, dtype=float32).to(device)
        test_target = torch.tensor(y_test, dtype=float32).to(device)

        #reduced_train_target = torch.tensor(y_train[pts], dtype=float32).to(device)

        # reduce features into isomap is its extensive
        if X_train.shape[-1] > 15:
            latent_len = 15
        else:
            latent_len = X_train.shape[-1]

        for r in range(run_number):
            print(f'{ds_name} - {r}')
            isomap_model = IsomapNN(reduced_dist, n_components=latent_len)
            isomap_model.to(device)
            reproj_features = isomap_model().to(float32)
            features = isomap_model.transform(dist_train_new)

            task_model = nn.Sequential(nn.Linear(latent_len, 512, dtype=float32),
                                       nn.Linear(512, 256, dtype=float32),
                                       nn.Linear(256, 64, dtype=float32),
                                       nn.Linear(64, 1, dtype=float32)
                                       ).to(device)
            task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
            task_criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()

            task_losses = []
            for ep in range(150):
                task_optim.zero_grad()
                out = task_model(features)
                task_loss = task_criterion(out, train_target.reshape_as(out))
                print(f'{r} - {ds_name} - epoch {ep}/{150}, loss={task_loss.item()}')
                task_losses.append(task_loss.item())
                task_loss.backward()
                task_optim.step()

            reproj_test_features = isomap_model.transform(test_dist)
            test_out = task_model(reproj_test_features)
            test_loss = task_criterion(test_out.reshape_as(test_target), test_target)
            train_loss = task_criterion(out.reshape_as(train_target), train_target)

            train_mae = mae_criterion(out.reshape_as(train_target), train_target)
            test_mae = mae_criterion(test_out.reshape_as(test_target), test_target)

            error_metric = pd.DataFrame()
            error_metric['train_mse'] = [train_loss.item()]
            error_metric['train_mae'] = [train_mae.item()]
            error_metric['test_mse'] = [test_loss.item()]
            error_metric['test_mae'] = [test_mae.item()]
            error_metric.to_csv(f'{ds_path}/{r}_isomap_metrics.csv', index=False)

            plot_predictoion_PCA_transform(X_train,
                                           features.cpu().detach().numpy(),
                                           y_train, out.cpu().detach().numpy(),
                                           f'MAE={train_mae.item()}, MSE={train_loss.item()}',
                                           f'{ds_path}/{r}_train.png')

            plot_predictoion_PCA_transform(X_test,
                                           reproj_test_features.cpu().detach().numpy(),
                                           y_test,
                                           test_out.cpu().detach().numpy(),
                                           f'MAE={test_mae.item()}, MSE={test_loss.item()}',
                                           f'{ds_path}/{r}_test.png')


run_isomap()