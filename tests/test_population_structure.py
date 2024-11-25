from itertools import combinations

import numpy as np
from copy import deepcopy

from tests.utils import create_connected_graph_individ
from evolution.PopulationEvoOperators import PopulationEvoOperators, PopulationMultiEvoOperators
from evolution.PopulationStructures import Population


def test_generate():
    base_individ = create_connected_graph_individ()
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
    base_individ = create_connected_graph_individ()
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
    base_individ = create_connected_graph_individ()
    population_shell = Population(size=5, base_individ=base_individ).generate()
    selected = [True, True, False, False, False]
    for i in range(5):
        population_shell.individs_pool[i].selected = selected[i]

    population_shell.individs_pool[1].adjacency_matrix = np.zeros_like(base_individ.adjacency_matrix)
    
    operator = PopulationEvoOperators(population=population_shell)
    operator.crossover_population()

    assert len(population_shell.individs_pool) == 7

    individs = np.array(population_shell.individs_pool)[[0,1,5,6]]
    index_for0 = 0
    index_for1 = 0
    j = 0

    for i in range(5):
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

        j += 1

    assert j < 5

    base_part0 = np.all(np.delete(np.delete(individs[0].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[index_for0].adjacency_matrix, i, axis=0), i, axis=1))
    base_part1 = np.all(np.delete(np.delete(individs[1].adjacency_matrix, i, axis=0), i, axis=1) == np.delete(np.delete(individs[index_for1].adjacency_matrix, i, axis=0), i, axis=1))

    assert base_part0 and base_part1
    assert np.all(individs[0].adjacency_matrix[i] == individs[index_for1].adjacency_matrix[i])
    assert np.all(individs[1].adjacency_matrix[i] == individs[index_for0].adjacency_matrix[i])


def test_filter_population():
    base_individ = create_connected_graph_individ()
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


def test_roulette_wheel_selection():
    base_individ = create_connected_graph_individ()
    population_shell = Population(size=5, base_individ=base_individ)
    population_shell = population_shell.generate()
    population_shell.individs_pool.append(deepcopy(base_individ))
    fitness = [1, 2, 1, 5, 1, 1.5]
    elitism = [False, False, False, True, False, False]

    for index in range(6):
        population_shell.individs_pool[index].fitness = fitness[index]
        population_shell.individs_pool[index].elitism = elitism[index]

    operator = PopulationEvoOperators(population=population_shell)
    operator.roulette_wheel_selection(winners_size=1)

    select_inds = [ind for ind in population_shell.individs_pool if ind.selected == True]

    assert len(select_inds) == 1

# multioptimization tests

def test_decomposition():
    '''
    Test, that checks sorting of individs in the population according to weight vectors (). Steps of test:

    1. Creating vectors (the weight vectors), size(vectors) == size_of_population
    2. Random mix indexes of individs and save in index_of_criteria
    3. Foreach index_of_criteria and assign half-hearted values of i weight vector to individ criteria 
    (it needs for that the criteria of individ were lying on the weight vector)
    4. Launch operator decomposition_population_by_vectors
    5. Check, that individs have an order in relation to the weight vectors 
    (i-individ criteria equal half-hearted i-weight vector)
    '''
    size_of_population = 5
    base_individ = create_connected_graph_individ()
    population_shell = Population(size=size_of_population, base_individ=base_individ)
    population_shell.generate()
    vectors = np.array([[0.  , 1.  ], [0.25, 0.75], [0.5 , 0.5 ], [0.75, 0.25], [1.  , 0.]])
    index_of_criteria = np.random.choice(size_of_population, size=size_of_population, replace=False)
    for i, value in enumerate(index_of_criteria):
        current_vector = vectors[value] / 2
        population_shell.individs_pool[i].criteria = current_vector

    population_operator = PopulationMultiEvoOperators(population=population_shell)
    population_operator.decomposition_population_by_vectors(vectors)

    result = True
    for i, individ in enumerate(population_shell.individs_pool):
        test_constr = np.all(individ.criteria == (vectors[i] / 2))
        result *= test_constr

    assert result

def test_check_dominante():
    base_individ = create_connected_graph_individ()
    one_individ = deepcopy(base_individ)
    two_individ = deepcopy(base_individ)
    population_shell = Population(size=2, base_individ=base_individ)

    population_operator = PopulationMultiEvoOperators(population=population_shell)

    one_individ.criteria = [1, 1]
    two_individ.criteria = [0, 1]
    one_individ.level = None
    two_individ.level = None

    population_shell.individs_pool = [one_individ, two_individ]

    assert population_operator.check_dominance(one_individ, two_individ) == False
    population_operator.fast_non_dominated_sorting()

    assert np.all([ind.level is not None for ind in population_shell.individs_pool])
    assert population_shell.individs_pool[1].level < population_shell.individs_pool[0].level

    one_individ.criteria = [0, 1]
    two_individ.criteria = [1, 1]
    one_individ.level = None
    two_individ.level = None

    population_shell.individs_pool = [one_individ, two_individ]

    assert population_operator.check_dominance(one_individ, two_individ) == True
    population_operator.fast_non_dominated_sorting()

    assert np.all([ind.level is not None for ind in population_shell.individs_pool])
    assert population_shell.individs_pool[0].level < population_shell.individs_pool[1].level

    one_individ.criteria = [0, 1]
    two_individ.criteria = [0, 1]
    one_individ.level = None
    two_individ.level = None

    population_shell.individs_pool = [one_individ, two_individ]

    assert population_operator.check_dominance(one_individ, two_individ) == False
    population_operator.fast_non_dominated_sorting()

    assert np.all([ind.level is not None for ind in population_shell.individs_pool])
    assert population_shell.individs_pool[0].level == population_shell.individs_pool[1].level

def test_selection_for_multiopt():
    base_individ = create_connected_graph_individ()
    population_shell = Population(size=5, base_individ=base_individ)
    population_shell.generate()

    population_operator = PopulationMultiEvoOperators(population=population_shell)
    population_operator.selection_for_multiopt(2)

    selected_individs = [individ for individ in population_shell.individs_pool if individ.selected]

    assert len(selected_individs) == 2
