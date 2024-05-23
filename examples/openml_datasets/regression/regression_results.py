import os

from matplotlib.colors import ListedColormap
from scipy import stats
import numpy as np
import openml
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter

log_folder = 'results/1000_20000_2024_05_19-15_42_20_PM'

datalist = openml.datasets.list_datasets(output_format="dataframe")


def return_samples_features_num(ds_id):
    d = datalist[datalist['did'] == int(ds_id)]
    return (d['NumberOfInstances'].values[0],
            d['NumberOfFeatures'].values[0],
            d['NumberOfNumericFeatures'].values[0],
            d['NumberOfSymbolicFeatures'].values[0],
            d['NumberOfClasses'].values[0])


def form_mean_table():
    labels = []
    pvals = []
    errors_df = pd.DataFrame()
    variances_df = pd.DataFrame()
    for folder in os.listdir(log_folder):
        if '.txt' not in folder:
            df = pd.read_csv(f'{log_folder}/{folder}/metrics.csv')
            df = df.drop(columns=['run'])

            if df.shape[0] != 0:
                pval = stats.friedmanchisquare(df['base_test_loss'], df['with_graph_test_loss'],
                                               df['with_evolution_test_loss'])
                pvals.append(pval[1])

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
                errors_df[folder] = normed_mean

                var_df = df.std(ddof=0) * 1.96
                variances_df[folder] = var_df / test_max

            else:
                print(f'Empty - {folder}')
    errors_df.style.apply(lambda col: ['font-weight:bold' if x == col.min() else '' for x in col])
    errors_df = errors_df.T
    errors_df.columns = df.columns
    variances_df = variances_df.T
    variances_df.columns = df.columns

    print(errors_df.to_string())

    num_of_samples_features = np.array([return_samples_features_num(i.split('_')[0]) for i in labels])

    errors_df['num_of_samples'] = num_of_samples_features[:, 0]
    errors_df['num_of_features'] = num_of_samples_features[:, 1]
    errors_df['num_of_num_features'] = num_of_samples_features[:, 2]
    errors_df['num_of_sym_features'] = num_of_samples_features[:, 3]
    errors_df['num_of_classes'] = num_of_samples_features[:, 4]
    errors_df['pvals'] = pvals

    variances_df['num_of_samples'] = num_of_samples_features[:, 0]
    variances_df['num_of_features'] = num_of_samples_features[:, 1]
    variances_df['num_of_num_features'] = num_of_samples_features[:, 2]
    variances_df['num_of_sym_features'] = num_of_samples_features[:, 3]
    variances_df['num_of_classes'] = num_of_samples_features[:, 4]
    variances_df['pvals'] = pvals

    plt.rcParams['figure.figsize'] = (8, 16)

    errors_df['size'] = errors_df['num_of_samples'] * errors_df['num_of_features']

    errors_df = errors_df.sort_values('num_of_samples')
    variances_df = variances_df.sort_values('num_of_samples')
    labels = errors_df.index.values.tolist()

    test_errors_df = errors_df[['base_test_loss', 'with_graph_test_loss', 'with_evolution_test_loss']].to_numpy()
    vars_errors_df = variances_df[['base_test_loss', 'with_graph_test_loss', 'with_evolution_test_loss']].to_numpy()
    b = np.zeros_like(test_errors_df)
    for i in range(b.shape[0]):
        if errors_df['pvals'][i] > 0.05 or np.isnan(errors_df['pvals'][i]):
            b[i, :] = 1
        else:
            if test_errors_df[i, 2] == np.min(test_errors_df[i]):
                b[i, :] = 2
            else:
                b[i, :] = 0

    # in case to plot mean values without variance
    # b = np.argsort(np.argsort(test_errors_df, axis=1), axis=1)

    # if not all classes are presented colors should be removed
    cmap = ListedColormap(["orange", "lightgrey", "palegreen"])
    im = plt.imshow(b, aspect="auto", cmap=cmap)

    # in case to plot mean values without variance
    # plt.colorbar(im, ticks=np.array([0.0, 0.5, 1.0]) * b.max(),
    # format=FixedFormatter(["low", "middle", "high"]), aspect=10)

    for i in range(test_errors_df.shape[0]):
        for j in range(test_errors_df.shape[1]):
            plt.text(j, i, f'{round(test_errors_df[i, j], 3)}+-{round(vars_errors_df[i, j], 3)}', ha="center",
                     va="center")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xticks(ticks=np.arange(3), labels=['No graph', 'Initial\ngraph', 'Evolution\ngraph'])
    plt.title('OpenML regression datasets\n50-20000 instances\nnormalized MAE ')
    plt.tight_layout()
    plt.savefig(f'{log_folder}/regr_openml_1000-20000.png', dpi=300)
    plt.show()


form_mean_table()
