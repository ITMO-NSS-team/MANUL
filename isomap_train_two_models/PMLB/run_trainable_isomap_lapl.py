import os.path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pmlb import fetch_data
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import nn, float32

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
        df = pd.read_csv(f'{save_path}/0_MY_CIRCLES.csv')
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

    X_validation = None
    y_validation = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # if dataset is large also make validation set as part of train
    if len(df) > 200:
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def laplac(graph):
    sigma = 1.0
    A = torch.exp(-graph ** 2 / (2 * sigma ** 2))
    D = torch.diag(torch.sum(A, axis=1))
    L = D - A
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    return L_sym

def train_isomap_with_init_cond(run_number=5):
    device = 'cuda'
    for ds_name in os.listdir('regression'):
        ds_path = f'regression/{ds_name}/isomap_linear_lapl'
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        X_train, X_test, X_validation, y_train, y_test, y_validation = get_data(ds_name,
                                                                                save_path=f'regression/{ds_name}')

        dist_train = torch.tensor(pairwise_distances(X_train, X_train), dtype=float32)
        dist_train = dist_train
        # SELECT SPARSE POINTS
        if X_train.shape[0] > 1000:
            retain_points = 1000
            pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)
        else:
            reduced_dist = torch.tensor(dist_train, dtype=float32).to(device)
            pts = np.arange(X_train.shape[0])

        dist_train_new = torch.tensor(pairwise_distances(X_train, X_train[pts]), dtype=float32).to(
            device)
        test_dist = torch.tensor(
            pairwise_distances(X_test, X_train[pts]), dtype=float32).to(device)

        # train_features = torch.tensor(X_train, dtype=float32).to(device)
        train_target = torch.tensor(y_train, dtype=float32).to(device)
        reduced_train_target = torch.tensor(y_train[pts], dtype=float32).to(device)
        # test_features = torch.tensor(X_test, dtype=float32).to(device)
        test_target = torch.tensor(y_test, dtype=float32).to(device)
        if X_validation is not None:
            val_dist = torch.tensor(pairwise_distances(X_validation, X_train[pts]), dtype=float32).to(
                device)
            # val_features = torch.tensor(X_validation, dtype=float32).to(device)
            val_target = torch.tensor(y_validation, dtype=float32).to(device)


        # reduce features into isomap is its extensive
        if X_train.shape[-1] > 15:
            latent_len = 15
        else:
            latent_len = X_train.shape[-1]

        for r in range(run_number):
            print(f'{ds_name} - {r}')

            # ________________TRAIN ISOMAP ______________________________
            isomap_model = IsomapNN(reduced_dist, n_components=latent_len)
            isomap_model.to(device)

            isomap_epochs = 1000
            task_epochs = 150
            isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.0001)
            isomap_criterion = nn.MSELoss()
            validation_criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            losses = []
            val_losses = []
            best_loss = np.inf
            best_val_loss = np.inf
            best_isomap_model = None

            # ISOMAP TRAIN LOOP
            for epoch in range(isomap_epochs):
                reproj_features = isomap_model().to(float32)

                with torch.no_grad():
                    features = isomap_model.transform(dist_train_new)
                    #graph = isomap_model.distances_matrix
                    #L_sym = laplac(graph)

                task_model = nn.Sequential(nn.Linear(latent_len, 512, dtype=float32),
                                  nn.Linear(512, 256, dtype=float32),
                                  nn.Linear(256, 64, dtype=float32),
                                  nn.Linear(64, 1, dtype=float32)
                                  ).to(device)
                task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                task_criterion = nn.MSELoss()

                task_losses = []
                for ep in range(task_epochs):
                    task_optim.zero_grad()
                    out = task_model(features)
                    model_loss = task_criterion(out.reshape_as(train_target), train_target)
                    #graph_loss = torch.dot(out[pts].reshape_as(train_target[pts]), L_sym@out[pts].reshape_as(train_target[pts]))
                    #task_loss = model_loss+0.1*graph_loss
                    #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}, '
                          #f'model_loss={model_loss.item()}, graph_loss={graph_loss}')
                    task_losses.append(model_loss.item())
                    model_loss.backward()
                    task_optim.step()

                output = task_model(reproj_features)
                isomap_loss = isomap_criterion(output.to(torch.float32),
                                               reduced_train_target.reshape_as(output).to(torch.float32))
                losses.append(isomap_loss.item())


                if X_validation is not None:
                    reproj_val_features = isomap_model.transform(val_dist)
                    val_output = task_model(reproj_val_features)
                    val_loss = validation_criterion(val_output.reshape_as(val_target), val_target)
                    val_losses.append(val_loss.item())

                    if val_losses[-1] < best_val_loss:
                        best_val_loss = val_losses[-1]
                        best_isomap_model = isomap_model

                if losses[-1] < best_loss:
                    best_loss = losses[-1]
                    if X_validation is None:
                        best_isomap_model = isomap_model

                isomap_loss.backward()
                isomap_optim.step()
                if X_validation is not None:
                    print(f'{ds_name} - epoch {epoch}/{isomap_epochs},  loss={losses[-1]}, val_loss={val_losses[-1]}')
                if X_validation is None:
                    print(f'{ds_name} - epoch {epoch}/{isomap_epochs},  loss={losses[-1]}')

                # ______________SAVE ISOMAP ON EUQLID DIST_______________________
                if epoch == 0:
                    reproj_test_features = isomap_model.transform(test_dist)
                    test_out = task_model(reproj_test_features)
                    test_loss = task_criterion(test_out.reshape_as(test_target), test_target)

                    train_mae = mae_criterion(out.reshape_as(train_target), train_target)
                    test_mae = mae_criterion(test_out.reshape_as(test_target), test_target)
                    if X_validation is not None:
                        val_mae = mae_criterion(val_output.reshape_as(val_target), val_target)

                    error_metric = pd.DataFrame()
                    error_metric['train_mse'] = [losses[-1]]
                    error_metric['train_mae'] = [train_mae.item()]
                    error_metric['test_mse'] = [test_loss.item()]
                    error_metric['test_mae'] = [test_mae.item()]

                    if X_validation is not None:
                        error_metric['val_mse'] = [val_loss.item()]
                        error_metric['val_mae'] = [val_mae.item()]

                    error_metric.to_csv(f'{ds_path}/{r}_isomap_raw_metrics.csv', index=False)

            plt.figure()
            plt.plot(np.arange(len(losses)), losses, label='Train')
            plt.axhline(best_loss, c='r', linestyle='dashed')
            plt.annotate(str(round(best_loss, 4)), (0, best_loss), c='r')

            if X_validation is not None:
                plt.plot(np.arange(len(val_losses)), val_losses, label='Validation')
                plt.axhline(best_val_loss, c='green', linestyle='dashed')
                plt.annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')

            plt.title('Convergence plot')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend()
            plt.tight_layout()
            plt.yscale('log')
            plt.savefig(f'{ds_path}/{r}_isomap_model_convergence.png')
            # plt.show()
            plt.close()

            # ISOMAP POINTS PROJECTION AFTER OPTIMIZATION
            train_reproj_points = best_isomap_model.transform(dist_train_new)
            test_proj_points = best_isomap_model.transform(test_dist)

            task_model = nn.Sequential(nn.Linear(latent_len, 512, dtype=float32),
                                  nn.Linear(512, 256, dtype=float32),
                                  nn.Linear(256, 64, dtype=float32),
                                  nn.Linear(64, 1, dtype=float32)
                                  ).to(device)
            task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
            task_criterion = nn.MSELoss()
            for ep in range(task_epochs):
                task_optim.zero_grad()
                out = task_model(features)
                task_loss = task_criterion(out.reshape_as(train_target), train_target)
                # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                task_loss.backward()
                task_optim.step()

            train_output = task_model(train_reproj_points.to(float32))
            test_output = task_model(test_proj_points.to(float32))

            if X_validation is not None:
                val_proj_points = best_isomap_model.transform(val_dist)
                val_output = task_model(val_proj_points.to(float32))

            # METRICS CALCULATION
            score = nn.MSELoss()

            train_loss = score(train_output.reshape_as(train_target), train_target)
            test_loss = score(test_output.reshape_as(test_target), test_target)
            train_mae = mae_criterion(train_output.reshape_as(train_target), train_target)
            test_mae = mae_criterion(test_output.reshape_as(test_target), test_target)

            error_metric = pd.DataFrame()
            error_metric['train_mse'] = [train_loss.item()]
            error_metric['train_mae'] = [train_mae.item()]
            error_metric['test_mse'] = [test_loss.item()]
            error_metric['test_mae'] = [test_mae.item()]
            if X_validation is not None:
                val_loss = score(val_output.reshape_as(val_target), val_target)
                val_mae = mae_criterion(val_output.reshape_as(val_target), val_target)
                error_metric['val_mse'] = [val_loss.item()]
                error_metric['val_mae'] = [val_mae.item()]
            error_metric.to_csv(f'{ds_path}/{r}_isomap_optimized_metrics.csv', index=False)

            train_proj_points = train_reproj_points.cpu().detach().numpy()
            test_proj_points = test_proj_points.cpu().detach().numpy()

            if X_validation is not None:
                val_proj_points = val_proj_points.cpu().detach().numpy()

            plot_predictoion_PCA_transform(points=X_train,
                                           proj_points=train_proj_points,
                                           true_labels=y_train,
                                           predicted_labels=train_output.cpu().detach().numpy(),
                                           title=f'MAE={train_mae}, MSE={train_loss}',
                                           save_path=f'{ds_path}/{r}_isomap_optimized_train.png')
            plot_predictoion_PCA_transform(points=X_test,
                                           proj_points=test_proj_points,
                                           true_labels=y_test,
                                           predicted_labels=test_output.cpu().detach().numpy(),
                                           title=f'MAE={test_mae}, MSE={test_loss}',
                                           save_path=f'{ds_path}/{r}_isomap_optimized_test.png')
            if X_validation is not None:
                plot_predictoion_PCA_transform(points=X_validation,
                                               proj_points=val_proj_points,
                                               true_labels=y_validation,
                                               predicted_labels=val_output.cpu().detach().numpy(),
                                               title=f'MAE={val_mae}, MSE={val_loss}',
                                               save_path=f'{ds_path}/{r}_isomap_optimized_validation.png')




train_isomap_with_init_cond()