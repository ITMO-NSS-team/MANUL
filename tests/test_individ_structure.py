import os
import sys

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy

from evolution.IndividStructures import DataStructureGraph

# def test_loss_function():
#     source_data_array = np.random.randint(0, 10, size=(8, 2))
#     adj = np.zeros((5, 5))
#     edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T
#     matrix = euclidean_distances(source_data_array[:5], source_data_array[:5]) 
#     adj[edges[0], edges[1]] = 1
#     adj[edges[1], edges[0]] = 1

#     individ_shell = DataStructureGraph()
#     individ_shell.source_data = source_data_array
#     individ_shell.adjacency_matrix = deepcopy(adj)
#     individ_shell.matrix_connect = deepcopy(matrix)
#     individ_shell.basis = np.arange(5)


# def test_add_edge():
#     n = 10
#     matrix = np.zeros((n, n), dtype=float)
#     individ_shell = DataStructureGraph()


def test_edge_len_mutation():
    n = 10
    matrix = np.zeros((n, n), dtype=float)

    individ_shell = DataStructureGraph()
    individ_shell.matrix_connect = matrix

    # change 6 edges
    edges_to_mutate_indices = np.random.randint(n, size=(6, 2))
    individ_shell.change_edges_length(edges_inds=edges_to_mutate_indices, mutate_intensity=0.3)
    plt.imshow(individ_shell.matrix_connect)
    plt.show()
    assert individ_shell.matrix_connect.diagonal().all() == 0
    assert np.all(np.abs(individ_shell.matrix_connect-individ_shell.matrix_connect.T) < 1e-8)
    assert individ_shell.matrix_connect[individ_shell.matrix_connect < 0].shape[0] == 0
    assert individ_shell.matrix_connect[individ_shell.matrix_connect > 1].shape[0] == 0


def test_subgraph_replacing():
    matrix = np.full((10, 10), 1)
    plt.imshow(matrix)
    plt.show()

    individ_shell = DataStructureGraph()
    individ_shell.adjacency_matrix = matrix
    edges_before = individ_shell.number_of_edges

    individ_shell.replace_subgraph(5, np.array([1, 2, 3]))
    edges_after = individ_shell.number_of_edges
    plt.imshow(individ_shell.adjacency_matrix)
    plt.show()

    assert edges_before - edges_after == 7

def test_twist_nodes():
    source_data_array = np.random.randint(0, 10, size=(8, 2))
    adj = np.zeros((5, 5))
    edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T
    matrix = euclidean_distances(source_data_array[:5], source_data_array[:5]) 
    adj[edges[0], edges[1]] = 1
    adj[edges[1], edges[0]] = 1

    individ_shell = DataStructureGraph()
    individ_shell.source_data = source_data_array
    individ_shell.adjacency_matrix = deepcopy(adj)
    individ_shell.matrix_connect = deepcopy(matrix)
    individ_shell.basis = np.arange(5)

    individ_shell.twist_nodes(np.array([0, 1, 2]))
    assert np.all(np.sort(individ_shell.basis[:3]) == np.array([5, 6, 7]))
    assert np.all(individ_shell.adjacency_matrix == adj)
    assert int(np.sum(individ_shell.matrix_connect != matrix) / 2) <= 3

# test_twist_nodes()