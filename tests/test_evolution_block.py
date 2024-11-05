import numpy as np
from copy import deepcopy
import os

from tests.utils import create_connected_graph_individ
from evolution.Evolution import Evolution, MultiEvolution
from regularizator.ModuleNN import ModelNN


def test_genearte_evolution():
    base_individ = create_connected_graph_individ()
    model = ModelNN(base_individ.source_data, np.random.randint(0, 2, size=base_individ.source_data.shape[0]),
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
    base_individ = create_connected_graph_individ()
    model = ModelNN(base_individ.source_data, np.random.randint(0, 2, size=base_individ.source_data.shape[0]),
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

def test_save_and_load_history():
    base_individ = create_connected_graph_individ()
    model = ModelNN(base_individ.source_data, np.random.randint(0, 2, size=base_individ.source_data.shape[0]),
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
    dict_for_test = {}
    for i in range(5):
        dict_for_test[i] = {
            "fitness": np.random.randint(10),
            "trained_loss_values": {
                "model_loss": np.random.random(),
                "graph_loss": np.random.random()
            }
        }
    
    evolution.save_history(dict_for_test, name="test_dict")
    assert os.path.exists(f'{base_individ.cache_folder}/test_dict.pkl')
    evolution.save_history(dict_for_test)
    assert os.path.exists(f'{base_individ.cache_folder}/history_from_evolution.pkl')

    population = deepcopy(evolution.population)
    population.individs_pool = []
    population.load_individs_pool(f'{base_individ.cache_folder}/test_dict.pkl')
    os.remove(f'{base_individ.cache_folder}/test_dict.pkl')
    os.remove(f'{base_individ.cache_folder}/history_from_evolution.pkl')

    assert len(population.individs_pool) == 5
    isEqual = True
    for i in range(5):
        isEqual *= population.individs_pool[i].fitness == dict_for_test[i]["fitness"]
        isEqual *= population.individs_pool[i].trained_loss_values == dict_for_test[i]["trained_loss_values"]

    assert isEqual


# tests for multioptimization

# def test_ini():
    