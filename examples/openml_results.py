import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter

log_folder = 'C:/Users/Julia/Documents/NSS_lab/fastnet/examples/openml_log/regression/2024_05_06-22_50_50_PM'


def save_boxplots():
    # список для списков из 3х разбросов для 5 запусков
    train_dataset_metrics = []
    test_dataset_metrics = []
    labels = []
    for folder in os.listdir(log_folder):
        '''plots_save_folder = f'{log_folder}/{folder}/boxplots'
        if not os.path.exists(plots_save_folder):
            os.mkdir(plots_save_folder)'''
        if '.txt' not in folder:
            df = pd.read_csv(f'{log_folder}/{folder}/metrics.csv')
            if df.shape[0] != 0:
                labels.append(folder)
                train_dataset_metrics.append([df['base_train_loss'].tolist(),
                                              df['with_graph_train_loss'].tolist(),
                                              df['with_evolution_train_loss'].tolist()])

                test_dataset_metrics.append([df['base_test_loss'].tolist(),
                                             df['with_graph_test_loss'].tolist(),
                                             df['with_evolution_test_loss'].tolist()])
            else:
                print(f'Empty - {folder}')

    for n in range(len(train_dataset_metrics)):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].boxplot(train_dataset_metrics[n][0:3], labels=['base', 'with graph', 'with evolution'])
        axs[0].set_title('MSE on train set')
        axs[1].boxplot(test_dataset_metrics[n][0:3], labels=['base', 'with graph', 'with evolution'])
        axs[1].set_title('MSE on test set')
        fig.suptitle(labels[n])
        # plt.savefig(f'{plots_save_folder}/')
        plt.show()


def form_mean_table():
    labels = []
    errors_df = pd.DataFrame()
    for folder in os.listdir(log_folder):
        if '.txt' not in folder:
            df = pd.read_csv(f'{log_folder}/{folder}/metrics.csv')
            df = df.drop(columns=['run'])
            if df.shape[0] != 0:
                labels.append(folder)
                mean_df = df.mean()
                train_max = df[['base_train_loss', 'with_graph_train_loss', 'with_evolution_train_loss']].max().max()
                test_max = df[['base_test_loss', 'with_graph_test_loss', 'with_evolution_test_loss']].max().max()

                normed_mean = []
                for i, val in enumerate(mean_df.values):
                    if i < 3:
                        normed_mean.append(val / train_max)
                    if i >= 3:
                        normed_mean.append(val / test_max)
                        # mean_df = mean_df/df.max()
                errors_df[folder] = normed_mean
            else:
                print(f'Empty - {folder}')
    errors_df.style.apply(lambda col: ['font-weight:bold' if x == col.min() else '' for x in col])
    errors_df = errors_df.T
    errors_df.columns = df.columns
    print(errors_df.to_string())

    train_errors_df = errors_df[['base_train_loss', 'with_graph_train_loss', 'with_evolution_train_loss']].to_numpy()
    b = np.argsort(np.argsort(train_errors_df, axis=1), axis=1)
    plt.rcParams['figure.figsize'] = (6, 16)
    im = plt.imshow(b, aspect="auto", cmap="Reds")
    plt.colorbar(im, ticks=np.array([0.0, 0.5, 1.0]) * b.max(),
                 format=FixedFormatter(["low", "middle", "high"]))
    for i in range(train_errors_df.shape[0]):
        for j in range(train_errors_df.shape[1]):
            plt.text(j, i, round(train_errors_df[i, j], 5), ha="center", va="center")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xticks(ticks=np.arange(3), labels=['base', 'with_graph', 'with_evolution'])
    plt.title('MSE/max(MSE) Train')
    plt.tight_layout()
    plt.show()

    test_errors_df = errors_df[['base_test_loss', 'with_graph_test_loss', 'with_evolution_test_loss']].to_numpy()
    b = np.argsort(np.argsort(test_errors_df, axis=1), axis=1)
    im = plt.imshow(b, aspect="auto", cmap="Reds")
    plt.colorbar(im, ticks=np.array([0.0, 0.5, 1.0]) * b.max(),
                 format=FixedFormatter(["low", "middle", "high"]))
    for i in range(test_errors_df.shape[0]):
        for j in range(test_errors_df.shape[1]):
            plt.text(j, i, round(test_errors_df[i, j], 5), ha="center", va="center")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xticks(ticks=np.arange(3), labels=['base', 'with_graph', 'with_evolution'])
    plt.title('MSE/max(MSE) Test')
    plt.tight_layout()
    plt.show()


form_mean_table()
