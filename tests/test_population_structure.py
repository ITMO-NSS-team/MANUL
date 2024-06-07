# import os
# import sys
from itertools import combinations

# root_dir = '/'.join(os.getcwd().split("/")[:-1])
# sys.path.append(root_dir)

import numpy as np
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
    matrix = matrix / np.max(matrix)
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
    
    combinations_individ = list(combinations(population_shell.individs_pool, 2))
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
        population_shell.individs_pool[index].fitness = fitness[index]

    operator = PopulationEvoOperators(population=population_shell)
    operator.elitism()

    elite_individ = [ind.elitism for ind in population_shell.individs_pool]
    elite_individ = np.where(elite_individ)[0]

    assert len(elite_individ) == 1
    assert elite_individ[0] == 3
    assert (population_shell.individs_pool[elite_individ[0]].elitism and population_shell.individs_pool[elite_individ[0]].selected)


def test_crossover_population():
    base_individ = create_base_individ()
    population_shell = Population(size=5, base_individ=base_individ).generate()
    selected = [True, True, False, False, False]
    for i in range(len(population_shell.individs_pool)):
        population_shell.individs_pool[i].selected = selected[i]

    population_shell.individs_pool[1].adjacency_matrix = np.zeros_like(base_individ.adjacency_matrix)
    
    operator = PopulationEvoOperators(population=population_shell)
    operator.crossover_population()

    assert len(population_shell.individs_pool) == 7

    individs = np.array(population_shell.individs_pool)[[0,1,5,6]]
    index_for0 = 0
    index_for1 = 0
    i = 0

    while i <= 4:
        test1 = np.all(np.delete(np.delete(individs[0].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[2].adjacency_matrix, i, axis=0), i, axis=1))
        test2 = np.all(np.delete(np.delete(individs[0].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[3].adjacency_matrix, i, axis=0), i, axis=1))

        if test1:
            index_for0 = 2
            index_for1 = 3
            break

        if test2:
            index_for0 = 3
            index_for1 = 2
            break

        i += 1

    assert i < 5

    base_part0 = np.all(np.delete(np.delete(individs[0].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[index_for0].adjacency_matrix, i, axis=0), i, axis=1))
    base_part1 = np.all(np.delete(np.delete(individs[1].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[index_for1].adjacency_matrix, i, axis=0), i, axis=1))

    assert base_part0 and base_part1
    assert np.all(individs[0].adjacency_matrix[i] == individs[index_for1].adjacency_matrix[i])
    assert np.all(individs[1].adjacency_matrix[i] == individs[index_for0].adjacency_matrix[i])


def test_filter_population():
    base_individ = create_base_individ()
    population_shell = Population(size=5, base_individ=base_individ)
    population_shell = population_shell.generate()
    population_shell.individs_pool.append(deepcopy(base_individ))
    fitness = [1, 2, 3, 5, 4, 1.5]
    elitism = [False, False, False, True, False, False]

    for index in range(6):
        population_shell.individs_pool[index].fitness = fitness[index]
        population_shell.individs_pool[index].elitism = elitism[index]

    operator = PopulationEvoOperators(population=population_shell)
    operator.filter_population(size_to_save=5)

    individs = population_shell.individs_pool
    elite_individs = [ind for ind in individs if ind.elitism]
    fitness_ind = [ind.fitness for ind in individs]
    assert len(individs) == 5
    assert len(elite_individs) == 1 and elite_individs[0].fitness == 5
    assert len(np.unique(fitness_ind)) == 5
    assert 1 not in fitness_ind
