import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy
import pickle as pkl
from torch import float64 as fl64
from torch import nn

from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN

def simple_nn(inp_dims):
    model = nn.Sequential(nn.Linear(inp_dims, 128, dtype=fl64),
                          nn.ReLU(),
                          nn.Linear(128, 1, dtype=fl64),
                          nn.ReLU())
    return model

def fake_loss(true, predicted):
    """
    Function to imitate the callable object of custom metric function
    """
    return 9999

def split_dataset(data, split_ratio=0.8):
    split_ratio = int(data.shape[0] * split_ratio)
    train = data[:split_ratio]
    test = data[split_ratio:]
    return train, test

def create_connected_graph_individ(source_data_array=None):
    if source_data_array is None:
        source_data_array = np.random.randint(0, 10, size=(8, 2))
    adj = np.zeros((5, 5))
    edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T
    matrix = euclidean_distances(source_data_array[:5], source_data_array[:5])
    matrix = matrix / np.max(matrix)
    adj[edges[0], edges[1]] = 1
    adj[edges[1], edges[0]] = 1

    individ_shell = DataStructureGraph()
    individ_shell.source_data = source_data_array
    individ_shell.adjacency_matrix = deepcopy(adj)
    individ_shell.matrix_connect = deepcopy(matrix)
    individ_shell.basis = np.arange(5)

    return individ_shell


def create_model_circle_withoutgraph():
    with open('tests/points_circle.pkl', 'rb') as f:
        features = pkl.load(f)

    features = np.array(sorted(features, key=lambda parameters: parameters[2]))
    target = np.linspace(0, 0.9, features.shape[0])

    model_structure = simple_nn(features.shape[1])

    model = ModelNN(model_structure=model_structure,
                    train_feature=features,
                    train_target=target,
                    criterion=nn.L1Loss(),
                    target_metric=fake_loss
                    )
    
    return model

def create_model_circle_withgraph():
    individ_shell = DataStructureGraph(graph_file='tests/graph_circle.pkl')

    with open('tests/points_circle.pkl', 'rb') as f:
        features = pkl.load(f)

    features = np.array(sorted(features, key=lambda parameters: parameters[2]))
    target = np.linspace(0, 0.9, features.shape[0])

    model_structure = simple_nn(features.shape[1])

    model = ModelNN(model_structure=model_structure,
                    train_feature=features,
                    train_target=target,
                    criterion=nn.L1Loss(),
                    target_metric=fake_loss
                    )
    
    return model, individ_shell, features