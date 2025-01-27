import os.path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pmlb import fetch_data
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch import nn, float32


def plot_predictoion_PCA_transform(points, true_labels, predicted_labels, mae_val, save_path):
    points_2d = PCA(n_components=2).fit_transform(points)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cs0 = axs[0].scatter(points_2d[:, 1], points_2d[:, 0], c=true_labels)
    fig.colorbar(cs0, ax=axs[0])
    axs[0].set_title('Target values')
    cs1 = axs[1].scatter(points_2d[:, 1], points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs1, ax=axs[1])
    axs[1].set_title('Predicted values')
    fig.suptitle(f'MAE={mae_val}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def mae(predicted, target):
    return np.sum(abs(predicted - target)) / predicted.shape[0]


def call_df_names(task):
    """
    task - "regression", "classification"
    """
    df = pd.read_csv('all_summary_stats.tsv', sep='\t')
    df = df[df['n_continuous_features'] >= 2]
    df = df[df['n_features'] == df['n_continuous_features']]
    df = df[df['task'] == task]
    datasets_names = df['dataset'].tolist()
    return datasets_names


def get_data(df_name, save_path=None):
    """
    normalize data per column
    """
    df = fetch_data(df_name)
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


def train_linear_regression(run_number=5):
    ds_names = call_df_names('regression')
    for ds_name in ds_names:
        print(ds_name)
        ds_path = f'regression/{ds_name}/linear_regression'
        if os.path.exists(ds_path):
            print(f'Skip {ds_name} - already calculated')

        # IF LINEAR REGRESSION WAS NOT CALCULATED
        if not os.path.exists(ds_path):

            try:
                X_train, X_test, X_validation, y_train, y_test, y_validation = get_data(ds_name,
                                                                                        save_path=f'regression/{ds_name}')
            except Exception as e:
                print(e)
                continue

            os.makedirs(ds_path)
            for r in range(run_number):
                print(f'{ds_name} - {r}')
                error_metric = pd.DataFrame()

                linear = LinearRegression().fit(X_train, y_train)
                train_pred = linear.predict(X_train)
                train_mae = mae(train_pred, y_train)
                plot_predictoion_PCA_transform(X_train, y_train, train_pred, train_mae,
                                               f'{ds_path}/{r}_Linear_reg_train.png')

                val_mae = None
                if X_validation is not None:
                    val_pred = linear.predict(X_validation)
                    val_mae = mae(val_pred, y_validation)
                    plot_predictoion_PCA_transform(X_validation, y_validation, val_pred, val_mae,
                                                   f'{ds_path}/{r}_Linear_reg_validation.png')

                test_pred = linear.predict(X_test)
                test_mae = mae(test_pred, y_test)
                plot_predictoion_PCA_transform(X_test, y_test, test_pred, test_mae, f'{ds_path}/{r}_Linear_reg_test.png')

                error_metric['train_mae'] = [train_mae]
                error_metric['val_mae'] = [val_mae]
                error_metric['test_mae'] = [test_mae]
                error_metric.to_csv(f'{ds_path}/{r}_Linear_reg_metrics.csv', index=False)


def train_linear_nn(run_number=5):
    device = 'cuda'
    #ds_names = call_df_names('regression')
    for ds_name in os.listdir('regression'):
        ds_path = f'regression/{ds_name}/linear_nn'
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        X_train, X_test, X_validation, y_train, y_test, y_validation = get_data(ds_name,
                                                                                save_path=f'regression/{ds_name}')
        train_features = torch.tensor(X_train, dtype=float32).to(device)
        train_target = torch.tensor(y_train, dtype=float32).to(device)
        test_features = torch.tensor(X_test, dtype=float32).to(device)
        test_target = torch.tensor(y_test, dtype=float32).to(device)
        if X_validation is not None:
            val_features = torch.tensor(X_validation, dtype=float32).to(device)
            val_target = torch.tensor(y_validation, dtype=float32).to(device)

        for r in range(run_number):
            print(f'{ds_name} - {r}')

            model = nn.Sequential(nn.Linear(X_train.shape[-1], 512, dtype=float32),
                                  nn.Linear(512, 256, dtype=float32),
                                  nn.Linear(256, 64, dtype=float32),
                                  nn.Linear(64, 1, dtype=float32)
                                  )
            model.to(device)
            optim = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
            criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            epochs = 150

            losses = []
            val_losses = []
            for ep in range(epochs):
                optim.zero_grad()
                out = model(train_features)
                loss = criterion(out.reshape_as(train_target), train_target)
                print(f'epoch {ep}/{epochs}, loss={loss.item()}')
                loss.backward()
                optim.step()
                losses.append(loss.item())

                if X_validation is not None:
                    val_out = model(val_features)
                    val_loss = criterion(val_out.reshape_as(val_target), val_target)
                    val_losses.append(val_loss.item())

            plt.plot(np.arange(len(losses)), losses, label='Train')
            if len(val_losses) > 0:
                plt.plot(np.arange(len(val_losses)), losses, label='Validation')
            plt.legend()
            plt.savefig(f'{ds_path}/{r}_convergence.png')
            plt.close()

            train_mae = mae_criterion(out.reshape_as(train_target), train_target)
            plot_predictoion_PCA_transform(X_train, y_train, out.cpu().detach().numpy(),
                                           train_mae.item(), f'{ds_path}/{r}_Linear_NN_train.png')

            if X_validation is not None:
                val_mae = mae_criterion(val_out.reshape_as(val_target), val_target)
                plot_predictoion_PCA_transform(X_validation, y_validation, val_out.cpu().detach().numpy(),
                                               val_mae.item(), f'{ds_path}/{r}_Linear_NN_validation.png')


            test_out = model(test_features)
            test_loss = criterion(test_out.reshape_as(test_target), test_target)
            test_mae = mae_criterion(test_out.reshape_as(test_target), test_target)
            plot_predictoion_PCA_transform(X_test, y_test, test_out.cpu().detach().numpy(),
                                           test_mae.item(), f'{ds_path}/{r}_Linear_NN_test.png')

            error_metric = pd.DataFrame()
            error_metric['train_mse'] = [losses[-1]]
            error_metric['train_mae'] = [train_mae.item()]
            error_metric['test_mse'] = [test_loss.item()]
            error_metric['test_mae'] = [test_mae.item()]

            if X_validation is not None:
                error_metric['val_mse'] = [val_losses[-1]]
                error_metric['val_mae'] = [val_mae.item()]

            error_metric.to_csv(f'{ds_path}/{r}_Linear_NN_metrics.csv', index=False)


train_linear_nn()