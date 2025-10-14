import os
from datetime import datetime

import numpy as np
import pickle
from matplotlib import pyplot as plt

from Eva.PopulationEvoOperators import PopulationEvoOperators
from Eva.IndividStructures import DataStructureGraph
from Eva.PopulationStructures import Population


class Evolution:
    def __init__(self,
                 train_features: np.ndarray,
                 train_targets: np.ndarray,
                 latent_len: int,
                 population_size: int,
                 iterations: int,
                 edges_mutation=True,
                 edges_weight_mutation=True,
                 logs_folder: [str, None] = None,
                 evo_operators_params: dict = None):
        """
        Class for setup and execution of the Eva optimization on graph to improve the model fit to data
        :param population_size: number of individs in population to produce from base_individ with mutation
        :param iterations: number of iterations for Eva
        :param evo_operators_params: dictionary with parameters for evolutionary operators for custom setting
        """
        self.best_individ = None
        self.evolution_history = None
        self.population_size = population_size
        self.iterations = iterations
        self.edges_mutation = edges_mutation
        self.edges_weight_mutation = edges_weight_mutation
        self._init_evo_operators_parameters(evo_operators_params)
        self.logs_folder = self._init_logs_folder(logs_folder)
        self.population = self.init_population(latent_len, targets=train_targets, features=train_features)

    def _init_logs_folder(self, folder: [str, None]):
        if folder is None:
            logs_folder = f"evolution_{datetime.now().strftime('%d%m%Y-%H.%M')}"
        else:
            logs_folder = folder
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        print(f'Logs folder set as: {logs_folder}')
        return logs_folder

    def _init_evo_operators_parameters(self, evo_operators_params: dict):
        self.evo_operators_params = {
            'elitism': {'elits_num': None},
            'rank_based_selection': {'tournament_size': None, 'winners_size': None},
            'crossover': {'crossover_size_percent': None},
            'mutation': {'mutation_prob': None}
        }
        if evo_operators_params:
            for operator in evo_operators_params.keys():
                for parameter in evo_operators_params[operator]:
                    self.evo_operators_params[operator][parameter] = evo_operators_params[operator][parameter]

    def init_population(self, latent_len, targets, features):
        individ = DataStructureGraph(dimensionality=latent_len, targets=targets, features=features)
        population = Population(base_individ=individ, size=self.population_size).generate(
            edges_mutation=self.edges_mutation,
            edges_weight_mutation=self.edges_weight_mutation)
        return population

    def evaluate_fitness(self):
        self.population = self.population.evaluate_individs_fitness()

    def plot_evolution_fitnesses(self, reverse: bool = False, save_path: str = None):
        ylab = 'Fitness'
        for generation in range(len(self.evolution_history.keys())):
            generation_fitnesses = [self.evolution_history[generation][g]['fitness'] for g in
                                    self.evolution_history[generation].keys()]
            if reverse:
                generation_fitnesses = 1 / np.array(generation_fitnesses)
                ylab = 'Loss'
            plt.scatter([generation] * len(self.evolution_history[generation]), generation_fitnesses)

        plt.title('Evolution convergence')
        plt.xlabel('Generation')
        plt.ylabel(ylab)
        if save_path is not None:
            plt.savefig(f'{save_path}')
            plt.close()
        else:
            plt.show()

    def run(self):
        evolution_history = {}
        best_individs_history = {}
        self.evaluate_fitness()
        self.population.visualize_population(f'{self.logs_folder}/iter_{0}.png')

        individ_parameters_dict = {}
        for k, individ in enumerate(self.population.individs_pool):
            individ_parameters_dict[k] = {'fitness': individ.fitness}
        evolution_history[0] = individ_parameters_dict

        for i in range(self.iterations):
            print(f'Evolution run, iteration - {i}')
            pop_operators = PopulationEvoOperators(population=self.population)

            print('Elite individs')
            pop_operators.elitism(elits_num=self.evo_operators_params["elitism"].
                                  get('elits_num', None))

            print('Selecting individs')
            pop_operators.rank_based_selection(
                tournament_size=self.evo_operators_params["rank_based_selection"].
                get('tournament_size', None),
                winners_size=self.evo_operators_params["rank_based_selection"].
                get('winners_size', None))

            print('Crossover individs')
            pop_operators.crossover_population(crossover_size_percent=self.evo_operators_params["crossover"].
                                               get('crossover_size_percent', None))

            print('Mutate individs')
            pop_operators.mutate_population(mutation_prob=self.evo_operators_params["mutation"].
                                            get('mutation_prob', None),
                                            edges_mutation=self.edges_mutation,
                                            edges_weight_mutation=self.edges_weight_mutation)

            print('Update fitnesses')
            self.evaluate_fitness()

            self.population.visualize_population(f'{self.logs_folder}/iter_{i + 1}.png')

            print('Filter population')
            pop_operators.filter_population(self.population_size)
            for individ in self.population.individs_pool:
                if individ.elitism: best_individs_history[i] = {"distances_matrix": individ.distances_matrix,
                                                                "fitness": individ.fitness,
                                                                "loss": individ.loss}
                individ.selected = False
                individ.elitism = False

            individ_parameters_dict = {}
            for k, individ in enumerate(self.population.individs_pool):
                individ_parameters_dict[k] = {'fitness': individ.fitness}

            evolution_history[i + 1] = individ_parameters_dict

        self.evolution_history = evolution_history

        best_individ_index = [ind.fitness for ind in self.population.individs_pool].index(
            max([ind.fitness for ind in self.population.individs_pool]))
        self.best_individ = self.population.individs_pool[best_individ_index]
        #self.save_history(best_individs_history, name='best_individs_by_iterations')

    def save_history(self, history: dict, name: str = None):
        if name is None:
            name = "history_from_evolution"

        with open(f'{self.logs_folder}/{name}.pkl', 'wb') as outp:
            pickle.dump(history, outp, pickle.HIGHEST_PROTOCOL)
            print(f'History Eva saved to {self.logs_folder}/{name}.pkl')
