# import os
# import sys

# root_dir = '/'.join(os.getcwd().split("/")[:-1])
# sys.path.append(root_dir)

import numpy as np
from copy import deepcopy

from evolution.IndividStructures import DataStructureGraph
from evolution.IndividEvoOperators import IndividEvoOperators

def create_base_individ():
    source_data_array = np.random.randint(0, 10, size=(8, 2))
    adj = np.zeros((5, 5))
    edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T
    matrix = []
    for elem in source_data_array[:5]:
        temp = []
        for elem2 in source_data_array[:5]:
            temp.append(np.sqrt(np.sum(np.power(elem - elem2, 2))))
        matrix.append(temp)
    matrix = np.array(matrix)
    matrix = matrix / np.max(matrix)
    adj[edges[0], edges[1]] = 1
    adj[edges[1], edges[0]] = 1

    individ_shell = DataStructureGraph()
    individ_shell.source_data = source_data_array
    individ_shell.adjacency_matrix = deepcopy(adj)
    individ_shell.matrix_connect = deepcopy(matrix)
    individ_shell.basis = np.arange(5)

    return individ_shell

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
    assert individ_shell.matrix_connect.diagonal().all() == 0
    assert np.all(np.abs(individ_shell.matrix_connect-individ_shell.matrix_connect.T) < 1e-8)
    assert individ_shell.matrix_connect[individ_shell.matrix_connect < 0].shape[0] == 0
    assert individ_shell.matrix_connect[individ_shell.matrix_connect > 1].shape[0] == 0


def test_subgraph_replacing():
    matrix = np.full((10, 10), 1)

    individ_shell = DataStructureGraph()
    individ_shell.adjacency_matrix = matrix
    edges_before = individ_shell.number_of_edges

    individ_shell.replace_subgraph(5, np.array([1, 2, 3]))
    edges_after = individ_shell.number_of_edges

    assert edges_before - edges_after == 7

def test_twist_nodes():
    source_data_array = np.random.randint(0, 10, size=(8, 2))
    adj = np.zeros((5, 5))
    edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T
    matrix = []
    for elem in source_data_array[:5]:
        temp = []
        for elem2 in source_data_array[:5]:
            temp.append(np.sqrt(np.sum(np.power(elem - elem2, 2))))
        matrix.append(temp)
    matrix = np.array(matrix) 
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

def test_mutate_base():
    individ_shell = create_base_individ()
    operator = IndividEvoOperators(individs=[individ_shell], base_mutation=True, edges_mutation=False, edges_weight_mutation=False)
    individs = operator.mutate(nodes_mutation_prob=0.1) # change 1 node

    assert len(individs) == 1
    assert individs[0].elitism == False and individs[0].fitness is None
    assert id(individs[0]) != id(individ_shell)
    assert len(np.unique(individs[0].basis)) == len(individ_shell.basis)
    assert np.sum(individ_shell.basis != individs[0].basis) == 1
    assert np.all(individs[0].adjacency_matrix == individ_shell.adjacency_matrix)
    assert np.all(individs[0].matrix_connect == individ_shell.matrix_connect)

def test_mutate_edge():
    individ_shell = create_base_individ()
    operator = IndividEvoOperators(individs=[individ_shell], base_mutation=False, edges_mutation=True, edges_weight_mutation=False)
    individs = operator.mutate(edges_existence_mutation_prob=0.3) 

    assert len(individs) == 1
    assert id(individs[0]) != id(individ_shell)
    assert individs[0].adjacency_matrix.diagonal().all() == 0
    assert np.any(individs[0].adjacency_matrix != individ_shell.adjacency_matrix)
    assert np.all(individs[0].basis == individ_shell.basis)
    assert np.all(individs[0].matrix_connect == individ_shell.matrix_connect)

def test_mutate_lenght():
    individ_shell = create_base_individ()
    operator = IndividEvoOperators(individs=[individ_shell], base_mutation=False, edges_mutation=False, edges_weight_mutation=True)
    individs = operator.mutate(edges_len_mutation_prob=0.4) # change 2 edges

    assert len(individs) == 1
    assert id(individs[0]) != id(individ_shell)
    assert individs[0].matrix_connect.diagonal().all() == 0
    assert np.sum(individs[0].matrix_connect != individ_shell.matrix_connect) // 2 == 2
    assert np.all(individs[0].basis == individ_shell.basis)
    assert np.all(individs[0].adjacency_matrix == individ_shell.adjacency_matrix)

def test_crossover_inidvids():
    individ_shell = create_base_individ()
    individ_shell1 = create_base_individ()
    individ_shell1.adjacency_matrix = np.zeros_like(individ_shell1.adjacency_matrix)

    operator_crossover = IndividEvoOperators(individs=[individ_shell, individ_shell1])
    new_individs = operator_crossover.crossover_individs()

    assert len(new_individs) == 2
    i = 0

    while i <= 4:
        base_part0 = np.all(np.delete(np.delete(individ_shell.adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(new_individs[0].adjacency_matrix, i, axis=0), i, axis=1))

        if base_part0:
            base_part1 = np.all(np.delete(np.delete(individ_shell1.adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(new_individs[1].adjacency_matrix, i, axis=0), i, axis=1))
            break

        i += 1

    assert i < 5
    assert base_part0 and base_part1
    assert np.all(individ_shell.adjacency_matrix[i] == new_individs[1].adjacency_matrix[i])
    assert np.all(individ_shell1.adjacency_matrix[i] == new_individs[0].adjacency_matrix[i])
