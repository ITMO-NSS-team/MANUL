import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy
import pickle as pkl

from tests.utils import create_connected_graph_individ
from evolution.IndividStructures import DataStructureGraph
from evolution.IndividEvoOperators import IndividEvoOperators

def test_properties():
    individ_shell = create_connected_graph_individ(source_data_array=np.array([[4, 0], [0, 0], [3, 9], [4, 2], [7, 7], [8, 2], [9, 0], [9, 9]]))

    res_laplacian = np.array([[ 1.17337028, -0.40406102,  0.        ,  0.        , -0.76930926],
       [-0.40406102,  1.36237587, -0.95831485,  0.        ,  0.        ],
       [ 0.        , -0.95831485,  1.67260056, -0.71428571,  0.        ],
       [ 0.        ,  0.        , -0.71428571,  1.3033008 , -0.58901509],
       [-0.76930926,  0.        ,  0.        , -0.58901509,  1.35832435]])
    
    assert np.all(np.isclose(individ_shell.laplacian, res_laplacian, atol=1e-8))
    assert individ_shell.number_of_nodes == 5
    assert individ_shell.number_of_edges == 5
    


def test_loss_function():
    individ_shell = create_connected_graph_individ(source_data_array=np.array([[4, 0], [0, 0], [3, 9], [4, 2], [7, 7], [8, 2], [9, 0], [9, 9]]))
    temp_target = np.array([1, 1, 1, 0, 0, 1, 1, 0])
    # took that indexes, because that indexs are existed in individ_shell.basis
    indexs_for_count = np.array([0, 1, 2])
    check_loss = np.dot(temp_target[indexs_for_count].T, individ_shell.laplacian[indexs_for_count][:, indexs_for_count])
    check_loss = np.dot(check_loss, temp_target[indexs_for_count])

    res_loss = individ_shell.loss_function(temp_target, np.append(indexs_for_count, 7))

    assert isinstance(res_loss, float)
    assert res_loss - check_loss < 1e-8
    


def test_manipulation_edge():
    n = 10
    matrix = np.zeros((n, n), dtype=float)
    individ_shell = create_connected_graph_individ()
    individ_shell.adjacency_matrix = matrix 
    init_eu_matrix = deepcopy(individ_shell.matrix_connect)
    edges = np.array([[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]).T

    individ_shell.add_edges(edges)

    assert np.sum(individ_shell.adjacency_matrix[edges[0], edges[1]]) == 5
    assert np.sum(individ_shell.adjacency_matrix[edges[1], edges[0]]) == 5
    assert np.all(init_eu_matrix == individ_shell.matrix_connect)

    individ_shell.remove_edges(edges)

    assert np.sum(individ_shell.adjacency_matrix) == 0
    assert np.all(init_eu_matrix == individ_shell.matrix_connect)


def test_edge_len_mutation():
    n = 10
    matrix = np.zeros((n, n), dtype=float)

    individ_shell = DataStructureGraph()
    individ_shell.matrix_connect = matrix

    # change 6 edges
    edges_to_mutate_indices = np.random.randint(n, size=(2, 6))
    individ_shell.change_edges_length(edges_inds=edges_to_mutate_indices, mutate_intensity=0.3)
    assert individ_shell.matrix_connect.diagonal().all() == 0
    assert np.sum(individ_shell.matrix_connect-individ_shell.matrix_connect.T) == 0
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
    individ_shell = create_connected_graph_individ()
    adj = deepcopy(individ_shell.adjacency_matrix)
    matrix = deepcopy(individ_shell.matrix_connect)
    inds = np.array([0, 1, 2])

    individ_shell.twist_nodes(inds)
    assert np.all(np.sort(individ_shell.basis[:3]) == np.array([5, 6, 7]))
    assert np.all(individ_shell.adjacency_matrix == adj)
    assert int(np.sum(individ_shell.matrix_connect != matrix) / 2) <= len(inds)

def test_mutate_base():
    individ_shell = create_connected_graph_individ()
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
    individ_shell = create_connected_graph_individ()
    operator = IndividEvoOperators(individs=[individ_shell], base_mutation=False, edges_mutation=True, edges_weight_mutation=False)
    individs = operator.mutate(edges_existence_mutation_prob=1)

    assert len(individs) == 1
    assert id(individs[0]) != id(individ_shell)
    assert individs[0].adjacency_matrix.diagonal().all() == 0
    assert np.any(individs[0].adjacency_matrix != individ_shell.adjacency_matrix)
    assert np.all(individs[0].basis == individ_shell.basis)
    assert np.all(individs[0].matrix_connect == individ_shell.matrix_connect)

def test_mutate_lenght():
    individ_shell = create_connected_graph_individ()
    operator = IndividEvoOperators(individs=[individ_shell], base_mutation=False, edges_mutation=False, edges_weight_mutation=True)
    individs = operator.mutate(edges_len_mutation_prob=0.4) # change 2 edges

    assert len(individs) == 1
    assert id(individs[0]) != id(individ_shell)
    assert individs[0].matrix_connect.diagonal().all() == 0
    assert np.sum(individs[0].matrix_connect != individ_shell.matrix_connect) // 2 == 2
    assert np.all(individs[0].basis == individ_shell.basis)
    assert np.all(individs[0].adjacency_matrix == individ_shell.adjacency_matrix)

def test_crossover_inidvids():
    individ_shell = create_connected_graph_individ()
    individ_shell1 = create_connected_graph_individ()
    individ_shell1.adjacency_matrix = np.zeros_like(individ_shell1.adjacency_matrix)

    operator_crossover = IndividEvoOperators(individs=[individ_shell, individ_shell1])

    new_individs = operator_crossover.crossover_individs()

    assert len(new_individs) == 2
    j = 0

    for i in range(5):
        base_part0 = np.all(np.delete(np.delete(individ_shell.adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(new_individs[0].adjacency_matrix, i, axis=0), i, axis=1))

        if base_part0:
            base_part1 = np.all(np.delete(np.delete(individ_shell1.adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(new_individs[1].adjacency_matrix, i, axis=0), i, axis=1))
            break
        j += 1

    assert j < 5
    assert base_part0 and base_part1
    assert np.all(individ_shell.adjacency_matrix[i] == new_individs[1].adjacency_matrix[i])
    assert np.all(individ_shell1.adjacency_matrix[i] == new_individs[0].adjacency_matrix[i])


def test_create_graph():
    individ_shell = DataStructureGraph()
    with open('tests/points_circle.pkl', 'rb') as f:
        points = pkl.load(f)

    individ_shell.create_graph(points)

    assert len(np.unique(individ_shell.basis)) == len(individ_shell.basis)
    assert individ_shell.adjacency_matrix.shape == individ_shell.matrix_connect.shape == (len(individ_shell.basis), len(individ_shell.basis))
    assert individ_shell.matrix_connect.max() <= 1 and individ_shell.matrix_connect.min() >= 0

def test_individ_cache():
    # creating individ with connected graph and checking saving in cache
    base_individ = create_connected_graph_individ()
    base_individ.save_cache_object()
    assert os.path.exists(f"{base_individ.cache_folder}/graph_obj.pkl")

    # creation empty individ
    individ_shell = DataStructureGraph(cache_folder="new_cache_individ")
    individ_shell.adjacency_matrix = None
    individ_shell.basis = None
    assert base_individ.adjacency_matrix is not None and base_individ.basis is not None

    # loading fields from first individ to second and checking that information had written
    individ_shell.load_cache_object(f"{base_individ.cache_folder}/graph_obj.pkl")
    assert np.all(individ_shell.adjacency_matrix == base_individ.adjacency_matrix) and np.all(individ_shell.basis == base_individ.basis)
    assert individ_shell.cache_folder != base_individ.cache_folder

    # updating information in second individ and saving only one field
    individ_shell.adjacency_matrix = None
    individ_shell.basis = None
    individ_shell.fitness = 5
    individ_shell.save_cache_object(name="another_cache", fields=["fitness"])
    assert os.path.exists(f"{individ_shell.cache_folder}/another_cache.pkl")

    # loading saved fields above to first individ and checking that other fields didn't change
    base_individ.load_cache_object(f"{individ_shell.cache_folder}/another_cache.pkl")
    assert base_individ.adjacency_matrix is not None and base_individ.basis is not None
    assert base_individ.fitness == individ_shell.fitness

    os.remove(f"{base_individ.cache_folder}/graph_obj.pkl")
    os.remove(f"{individ_shell.cache_folder}/another_cache.pkl")