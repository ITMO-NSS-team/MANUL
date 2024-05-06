import os

import openml
import pandas as pd

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN
from sklearn.model_selection import train_test_split

log_folder = f'C:/Users/Julia/Documents/NSS_lab/fastnet/examples/openml_log/regression/2024_05_02-12_51_47_PM'

def split_train_test(dataset, target_name, split_ratio: float = 0.2):
    y = dataset[target_name].to_numpy()
    X = dataset[dataset.columns.drop(target_name)].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0)
    return X_train, y_train, X_test, y_test


for folder in os.listdir(log_folder):
    if '.txt' not in folder:
        df = pd.read_csv(f'{log_folder}/{folder}/metrics.csv')
        if df.shape[0] == 0:
            print(f'Process {folder}')
            id = int(folder.split('_')[0])
            dataset = openml.datasets.get_dataset(id)
            dataset_name = dataset.name
            target_name = dataset.default_target_attribute
            dataset_df = dataset.get_data()[0]
            dataset_df = dataset_df.apply(pd.to_numeric, errors='coerce')
            dataset_df = dataset_df.dropna()

            X_train, y_train, X_test, y_test = split_train_test(dataset_df, target_name)
            base_individ = DataStructureGraph(data=X_train,
                                              cash_folder=f'{log_folder}/{folder}',
                                              graph_file='base_graph.pkl')
            with_evolution_model = ModelNN(X_train[base_individ.basis], y_train[base_individ.basis],
                                           num_epochs=100,
                                           batch_size=300,
                                           problem='regres',
                                           cash_folder=f'C:/Users/Julia/Documents/NSS_lab/fastnet/examples/openml_log/test',
                                           model_name='with_evolution'
                                           )

            evolution = Evolution(base_individ=base_individ,
                                  iterations=5,
                                  population_size=10,
                                  model_to_optimize=with_evolution_model)
            evolution.run()
