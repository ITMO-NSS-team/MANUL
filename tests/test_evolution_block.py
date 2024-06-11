import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationStructures import Population
from regularizator.ModuleNN import ModelNN

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

def test_genearte_evolution():
    base_individ = create_base_individ()
    model = ModelNN(base_individ.source_data[base_individ.basis], np.random.randint(0, 2, size=len(base_individ.basis)),
                               num_epochs=10,
                               batch_size=10,
                               problem='binary_class')
    
    evo_operators = {
            'elitism': {'elits_num': 1},
            'roulette_wheel_selection': {'tournament_size': 10},
            'crossover': {'crossover_size_percent': 0.3},
            'mutation': {'mutation_prob': 0.5}
        }
    
    evolution = Evolution(base_individ=base_individ, population_size=3, iterations=1, model_to_optimize=model, evo_operators_params=evo_operators, edges_mutation=False)

    assert evolution.population_size == 3 and evolution.population.size == 3 and len(evolution.population.individs_pool) == 3
    assert id(evolution.base_model) == id(model)
    results = []
    for key in evo_operators:
        for key1 in evo_operators[key]:
            results.append(evolution.evo_operators_params[key].get(key1) == evo_operators[key].get(key1, None))
    
    assert np.all(results)
    assert id(evolution.base_individ) == id(base_individ)
    assert evolution.iterations == 1
    assert np.all([evolution.base_mutation, evolution.edges_mutation, evolution.edges_weight_mutation] == [True, False, True])

def test_evaluate_fitness():
    base_individ = create_base_individ()
    model = ModelNN(base_individ.source_data[base_individ.basis], np.random.randint(0, 2, size=len(base_individ.basis)),
                               num_epochs=10,
                               batch_size=10,
                               problem='binary_class')
    
    model_for_test = deepcopy(model)
    
    evo_operators = {
            'elitism': {'elits_num': 1},
            'roulette_wheel_selection': {'tournament_size': 10},
            'crossover': {'crossover_size_percent': 0.3},
            'mutation': {'mutation_prob': 0.5}
        }
    
    evolution = Evolution(base_individ=base_individ, population_size=3, iterations=1, model_to_optimize=model, evo_operators_params=evo_operators, edges_mutation=False)
    evolution.evaluate_fitness()

    results = []
    for p1, p2 in zip(evolution.base_model.model.parameters(), model_for_test.model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            results.append(False)
        results.append(True)

    assert np.all(list(map(lambda elem: elem is not None, [evo_elem.fitness for evo_elem in evolution.population.individs_pool])))
    assert np.all(results)
