import os
from datetime import datetime

import numpy as np
import openml
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN

def split_train_test_shuffle(dataset_df, target_name):
    y = dataset_df[target_name].to_numpy()
    X = dataset_df[dataset_df.columns.drop(target_name)].to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_idx, test_idx in splitter.split(X, y):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def run_openml_binary_classification(n_runs=5):
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')
    exp_folder = f'results/binary_50_1000_{start_time}'
    os.mkdir(exp_folder)
    log_file = f'{exp_folder}/log.txt'

    datalist = openml.datasets.list_datasets(output_format="dataframe")
    datalist['ValidInstNum'] = datalist['NumberOfInstances'] - datalist['NumberOfInstancesWithMissingValues']
    datasets_list = datalist[
        (datalist['NumberOfClasses'] == 2) & (datalist['ValidInstNum'] < 1000) & (datalist['ValidInstNum'] > 50)]

    for id in datasets_list['did']:
        try:
            dataset = openml.datasets.get_dataset(id)
            dataset_name = dataset.name
            target_name = dataset.default_target_attribute
            dataset_df = dataset.get_data()[0]
            for column in dataset_df.columns:
                if dataset_df[column].dtype.name in ['category', 'object']:
                    encoder = OneHotEncoder()
                    encoder.fit_transform(dataset_df[column].to_frame())
                    dataset_df[column] = encoder.transform(dataset_df[column].to_frame()).toarray()
            dataset_df = dataset_df.dropna()

            with open(log_file, 'a') as file:
                file.write(f"######\n\n"
                           f"dataset_num {id}\n"
                           f"dataset_name {dataset_name}\n"
                           f"rows_num {dataset_df.shape[0]}\n"
                           f"cols_num {dataset_df.shape[1]}\n\n")

            X_train, y_train, X_test, y_test = split_train_test_shuffle(dataset_df, target_name)

            ds_folder = f'{exp_folder}/{id}_{dataset_name}'
            if not os.path.exists(ds_folder):
                os.makedirs(ds_folder)

            dataset_df.to_csv(f'{ds_folder}/{dataset_name}.csv')
            metrics_file = f'{ds_folder}/metrics.csv'
            with open(metrics_file, 'a') as file:
                file.write(f"run,base_train_loss,with_graph_train_loss,with_evolution_train_loss,"
                           f"base_test_loss,with_graph_test_loss,with_evolution_test_loss\n")

            base_individ = DataStructureGraph(data=X_train,
                                              cash_folder=ds_folder)
            base_individ.show_2d(y_train, save_path=f'{ds_folder}/base_graph.png')

            with open(log_file, 'a') as file:
                file.write(f"base_graph_nodes {base_individ.number_of_nodes}\n"
                           f"base_graph_edges {base_individ.number_of_edges}\n\n")

            for n in range(n_runs):
                try:
                    r = str(n)
                    if not os.path.exists(f'{ds_folder}/{r}'):
                        os.makedirs(f'{ds_folder}/{r}')
                    with open(log_file, 'a') as file:
                        file.write(f"run_number {r}\n\n")
                    base_individ = DataStructureGraph(data=X_train,
                                                      cash_folder=ds_folder,
                                                      graph_file='base_graph.pkl')
                    base_individ.cash_folder = f'{ds_folder}/{r}'

                    base_model = ModelNN(X_train[base_individ.basis], y_train[base_individ.basis],
                                         num_epochs=150,
                                         batch_size=300,
                                         problem='binary_class',
                                         cash_folder=f'{ds_folder}/{r}',
                                         model_name='base_model')
                    base_model.train()
                    base_train_loss = base_model.get_metric_on_train()
                    base_test_loss = base_model.get_metric_on_test(X_test, y_test)

                    with open(log_file, 'a') as file:
                        file.write(f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')}\n"
                                   f"base_train_loss {base_train_loss}\n"
                                   f"base_test_loss {base_test_loss}\n")

                    with_graph_model = ModelNN(X_train, y_train,
                                               num_epochs=150,
                                               batch_size=300,
                                               problem='binary_class',
                                               cash_folder=f'{ds_folder}/{r}',
                                               model_name='with_graph'
                                               )
                    with_graph_model.train(base_individ)
                    with_graph_train_loss = with_graph_model.get_metric_on_train()
                    with_graph_test_loss = with_graph_model.get_metric_on_test(X_test, y_test)

                    with open(log_file, 'a') as file:
                        file.write(f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')}\n"
                                   f"with_graph_train_loss {with_graph_train_loss}\n"
                                   f"with_graph_test_loss {with_graph_test_loss}\n")

                    with_evolution_model = ModelNN(X_train, y_train,
                                                   num_epochs=150,
                                                   batch_size=300,
                                                   problem='binary_class',
                                                   cash_folder=f'{ds_folder}/{r}',
                                                   model_name='with_evolution'
                                                   )

                    evolution = Evolution(base_individ=base_individ,
                                          iterations=30,
                                          population_size=10,
                                          model_to_optimize=with_evolution_model)
                    evolution.run()
                    evolution.base_individ.show_2d(y_train, save_path=f'{ds_folder}/{r}/final_graph.png')
                    evolution.plot_evolution_fitnesses(save_path=f'{ds_folder}/{r}/evolution_conv.png')

                    with_evolution_train_loss = with_evolution_model.get_metric_on_train()
                    with_evolution_test_loss = with_evolution_model.get_metric_on_test(X_test, y_test)

                    with open(log_file, 'a') as file:
                        file.write(f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')}\n"
                                   f"with_evolution_train_loss {with_evolution_train_loss}\n"
                                   f"with_evolution_test_loss {with_evolution_test_loss}\n\n")

                    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                    axs[0].bar(['base', 'with graph', 'with evolution'],
                               [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
                    axs[0].set_title('ROC AUC on train set')
                    axs[1].bar(['base', 'with graph', 'with evolution'],
                               [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
                    axs[1].set_title('ROC AUC on test set')
                    plt.savefig(f'{ds_folder}/{r}/metrics_bar.png')
                    plt.close()

                    with open(metrics_file, 'a') as file:
                        file.write(f"{r},{base_train_loss},{with_graph_train_loss},{with_evolution_train_loss},"
                                   f"{base_test_loss},{with_graph_test_loss},{with_evolution_test_loss}\n")

                except Exception as e:
                    with open(log_file, 'a') as file:
                        file.write(f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')}\n"
                                   f"{e}\n")
                    continue

        except Exception as e:
            with open(log_file, 'a') as file:
                file.write(f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')}\n"
                           f"{e}\n")
            continue

run_openml_binary_classification()