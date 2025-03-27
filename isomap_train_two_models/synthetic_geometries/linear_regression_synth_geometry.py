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

from isomap_train_two_models.synthetic_geometries.data_generation import geometries
import plotly.express as px

def plot_3d_html(x, y, z, colors, path_to_save):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['colors'] = colors
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='colors')
    fig.update_scenes(aspectmode='data')
    fig.write_html(path_to_save)


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


def plot_3d_prediction(points, true_labels, predicted_labels, mae_val, save_path):
  fig = plt.figure(figsize=(12, 5))
  ax1 = fig.add_subplot(1, 2, 1, projection='3d')
  ax2 = fig.add_subplot(1, 2, 2, projection='3d')

  ax1.scatter(points[:, 1], points[:, 0], points[:, 2], c=true_labels)
  ax1.set_title('Target')
  ax2.scatter(points[:, 1], points[:, 0], points[:, 2], c=predicted_labels)
  ax2.set_title('Prediction')
  plt.suptitle(f'MAE={mae_val}')
  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()


def mae(predicted, target):
    return np.sum(abs(predicted - target)) / predicted.shape[0]


def mse(predicted, target):
    return np.sum((predicted - target)**2) / predicted.shape[0]


def train_linear_regression(run_number=5):
    for geometry in geometries.keys():

        data, labels = geometries[geometry]()

        X_train, X_test, y_train, y_test = train_test_split(data, labels)

        geom_path = f'results_(3k_1var)/{geometry}/linear_regression'
        if not os.path.exists(geom_path):
            os.makedirs(geom_path)
        for r in range(run_number):
            print(f'{geom_path} - {r}')
            error_metric = pd.DataFrame()

            linear = LinearRegression().fit(X_train, y_train)
            train_pred = linear.predict(X_train)
            train_mae = mae(train_pred, y_train)
            train_mse = mse(train_pred, y_train)
            plot_predictoion_PCA_transform(X_train, y_train, train_pred, train_mae,
                                           f'{geom_path}/{r}_Linear_reg_train.png')
            plot_3d_prediction(X_train, y_train, train_pred, train_mae,
                                           f'{geom_path}/{r}_Linear_reg_train_3d.png')
            plot_3d_html(X_train[:, 0], X_train[:, 1], X_train[:, 2], y_train, f'{geom_path}/train_target.html')
            plot_3d_html(X_train[:, 0], X_train[:, 1], X_train[:, 2], train_pred, f'{geom_path}/train_prediction.html')

            test_pred = linear.predict(X_test)
            test_mae = mae(test_pred, y_test)
            test_mse = mse(test_pred, y_test)
            plot_predictoion_PCA_transform(X_test, y_test, test_pred, test_mae, f'{geom_path}/{r}_Linear_reg_test.png')
            plot_3d_prediction(X_test, y_test, test_pred, test_mae, f'{geom_path}/{r}_Linear_reg_test_3d.png')

            error_metric['train_mae'] = [train_mae]
            error_metric['train_mse'] = [train_mse]
            error_metric['test_mae'] = [test_mae]
            error_metric['test_mse'] = [test_mse]
            error_metric.to_csv(f'{geom_path}/{r}_Linear_reg_metrics.csv', index=False)

train_linear_regression()