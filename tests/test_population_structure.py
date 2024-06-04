import os
import sys
from itertools import combinations

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy

from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationEvoOperators import PopulationEvoOperators
from evolution.PopulationStructures import Population

def create_base_individ():
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

    return individ_shell

def test_generate():
    base_individ = create_base_individ()
    population_shell = Population(size=5, base_individ=base_individ)
    population_shell = population_shell.generate()

    assert len(np.unique([id(elem) for elem in population_shell.individs_pool])) == 5
    
    combinations_individ = combinations(population_shell.individs_pool, 2)
    results =  np.full(len(combinations_individ), False)
    i = 0
    for individ1, individ2 in combinations_individ:
        results[i] = np.all([np.all([individ1.basis == individ2.basis]), np.all([individ1.adjacency_matrix == individ2.adjacency_matrix]), np.all([individ1.matrix_connect, individ2.matrix_connect])])
        i += 1

    assert np.any(results) == False


def test_elitism():
    base_individ = create_base_individ()
    population_shell = Population(size=5, base_individ=base_individ)
    population_shell = population_shell.generate()
    fitness = [2, 1, 3, 5, 4]

    for index in range(5):
        population_shell[index].fitness = fitness[index]

    operator = PopulationEvoOperators(population=population_shell)
    operator.elitism()

    elite_individ = [ind.elitism for ind in population_shell.individs_pool]
    elite_individ = np.where(elite_individ)

    assert len(elite_individ) == 1
    assert elite_individ[0] == 3
    assert (population_shell.individs_pool[elite_individ[0]].elitism and population_shell.individs_pool[elite_individ[0]].selected)


def test_crossover_population():
    base_individ = create_base_individ()
    population_shell = Population(size=5, base_individ=base_individ).generate()
    selected = [False, True, True, False, False]
    for i in range(len(population_shell.individs_pool)):
        population_shell.individs_pool[i].selected = selected[i]
    
    operator = PopulationEvoOperators(population=population_shell)
    operator.crossover_population()

    assert len(population_shell) == 7

    index_for1 = np.argmax([np.sum(population_shell.individs_pool[1].adjacency_matrix == population_shell.individs_pool[5].adjacency_matrix), np.sum(population_shell.individs_pool[1].adjacency_matrix == population_shell.individs_pool[6].adjacency_matrix)])
    index_for2 = 6 - index_for1
    index_for1 = 5 + index_for1

    replace_node1 = [not np.all(node) for node in  population_shell.individs_pool[1].adjacency_matrix == population_shell.individs_pool[index_for1].adjacency_matrix]
    replace_node1 = np.where(replace_node1 == True)[0]
    replace_node2 = [not np.all(node) for node in  population_shell.individs_pool[2].adjacency_matrix == population_shell.individs_pool[index_for2].adjacency_matrix]
    replace_node2 = np.where(replace_node2 == True)[0]

    assert len(replace_node1) == 1
    assert replace_node1 == replace_node2
    assert np.all(population_shell.individs_pool[1].adjacency_matrix[replace_node1[0]] == population_shell.individs_pool[index_for2].adjacency_matrix[replace_node1[0]])
    assert np.all(population_shell.individs_pool[2].adjacency_matrix[replace_node1[0]] == population_shell.individs_pool[index_for1].adjacency_matrix[replace_node1[0]])
    
test_generate()