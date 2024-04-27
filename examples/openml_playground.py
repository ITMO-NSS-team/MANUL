import numpy as np
import openml
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def get_dataset(dataset_num):
    dataset = openml.datasets.get_dataset(dataset_num).get_data()[0]
    y = dataset['Class'].to_numpy()

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    X = dataset[dataset.columns.drop('Class')].to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_idx, test_idx in splitter.split(X, y):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)



def run_experiment(dataset_index):
    X_train, y_train, X_test, y_test = get_dataset(dataset_index)

    base_individ = DataStructureGraph(data=X_train,
                                      n_neighbors=10,
                                      eps=0.15,
                                      cash_folder=f'C:/Users/Julia/Documents/NSS_lab/fastnet/examples/openml_log/{dataset_index}')

    base_individ.show_3d(labels=y_train, title='Before evolution')
    base_individ.show_2d(labels=y_train, euclidean=True)

    # считаем для простой нейронки без графа
    base_model = ModelNN(X_train[base_individ.basis], y_train[base_individ.basis],
                         num_epochs=50,
                         batch_size=300, problem='class')
    base_model.train()
    base_train_loss = base_model.get_loss_on_train()
    base_test_loss = base_model.get_loss_on_test(X_test, y_test)

    # считаем для простой нейронки с базовым графом
    with_graph_model = ModelNN(X_train[base_individ.basis], y_train[base_individ.basis],
                               num_epochs=50,
                               batch_size=300, problem='class')
    with_graph_model.train(base_individ)
    with_graph_train_loss = with_graph_model.get_loss_on_train()
    with_graph_test_loss = base_model.get_loss_on_test(X_test, y_test)

run_experiment(1471)