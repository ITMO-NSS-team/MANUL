import os
from datetime import datetime

import numpy as np
import pickle
from matplotlib import pyplot as plt

from evolution.PopulationEvoOperators_new import PopulationEvoOperators, PopulationMultiEvoOperators
from evolution.IndividStructures_new import DataStructureGraph
from evolution.PopulationStructures_new import Population


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
        Class for setup and execution of the evolution optimization on graph to improve the model fit to data
        :param population_size: number of individs in population to produce from base_individ with mutation
        :param iterations: number of iterations for evolution
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
            'roulette_wheel_selection': {'tournament_size': None, 'winners_size': None},
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
            pop_operators.roulette_wheel_selection(
                tournament_size=self.evo_operators_params["roulette_wheel_selection"].
                get('tournament_size', None),
                winners_size=self.evo_operators_params["roulette_wheel_selection"].
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

        # overwrite base_individ and base model to best individ
        best_individ_index = [ind.fitness for ind in self.population.individs_pool].index(
            max([ind.fitness for ind in self.population.individs_pool]))
        self.best_individ = self.population.individs_pool[best_individ_index]
        self.save_history(best_individs_history, name='best_individs_by_iterations')

    def save_history(self, history: dict, name: str = None):
        if name is None:
            name = "history_from_evolution"

        with open(f'{self.logs_folder}/{name}.pkl', 'wb') as outp:
            pickle.dump(history, outp, pickle.HIGHEST_PROTOCOL)
            print(f'History evolution saved to {self.logs_folder}/{name}.pkl')


class MultiEvolution(Evolution):
    def __init__(self, base_individ: DataStructureGraph,
                 population_size: int,
                 iterations: int,
                 base_mutation=True,
                 edges_mutation=True,
                 edges_weight_mutation=True,
                 evo_operators_params: dict = None):

        """
        Class for set and run multi-objective evolutionary optimization on graph for best model fitting on data
        :param base_individ: DataStructureGraph class object as the starting point
        :param population_size: number of individs in population to produce from base_individ with mutation
        :param iterations: number of iterations for evolution
        :param evo_operators_params: dictionary with parameters for evolutionary operators for custom setting
        """

        super().__init__(base_individ, population_size, iterations, base_mutation, edges_mutation,
                         edges_weight_mutation, evo_operators_params)

        self.__init_weights_vector()

    def __init_weights_vector(self):
        x = np.linspace(0, 1, self.population_size)
        y = x[-1::-1]
        self.weights_vector = np.array([x, y]).T

    def plot_vectors(self, path=None):
        '''
        Drawing weigth vectors for demonstration of Pareto-front for individs (primarily for debug purposes).
        :param path: path for saving image. Optional, by default image will showed.
        '''
        points = []
        ind_from_pop = self.population.individs_pool
        for i in range(self.population_size):
            plt.plot([0, self.weights_vector[i][0]], [0, self.weights_vector[i][1]])
            points.append(ind_from_pop[i].criteria)
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1])
        if path is not None:
            plt.savefig(path)
            plt.clf()
        else:
            plt.show()

    def run(self):
        evolution_history = {}
        self.best_individs_history = {}
        self.evaluate_criteria()

        self.best_individs_history[0] = self.population.individs_pool

        for i in range(self.iterations):
            print(f'Evolution run, iteration - {i}')
            pop_operators = PopulationMultiEvoOperators(population=self.population)
            pop_operators.decomposition_population_by_vectors(self.weights_vector)
            self.plot_vectors(path=f"{self.population.individs_pool[0].cache_folder}/vector{i}.png")

            for idx_vector, vector in enumerate(self.weights_vector):
                print('Search non-dominant individs')
                pop_operators.fast_non_dominated_sorting()
                print('Selecting individs')  # - получается первый индивид
                pop_operators.selection_for_multiopt(index_vector=idx_vector)
                print('Crossover')
                pop_operators.crossover_population(crossover_size_percent=self.evo_operators_params["crossover"].
                                                   get('crossover_size_percent', None))
                print('Mutate')
                pop_operators.mutate_population(mutation_prob=self.evo_operators_params["mutation"].
                                                get('mutation_prob', None),
                                                base_mutation=self.base_mutation,
                                                edges_mutation=self.edges_mutation,
                                                edges_weight_mutation=self.edges_weight_mutation)
                print('Count criteria in new individs')
                self.evaluate_criteria()
                print('Replace individs')
                pop_operators.form_popualtion_with_new_individs()
                for individ in self.population.individs_pool:
                    individ.selected = False

            self.best_individs_history[i + 1] = self.population.individs_pool

        self.save_individs()

    def save_individs(self):
        '''
        Saving all individs at the current moment.
        '''
        for index, individ in enumerate(self.population.individs_pool):
            individ.save_cache_object(name=f'final_graph{index}')

    def evaluate_criteria(self):
        self.population = self.population.evaluate_individs_criteria()

    def plot_loss_best_individ(self):
        """
        Plotting loss for the best individ in population for each iteration of evolution.
        """
        ylab = 'MSE Loss'

        metrics = []
        for iter in self.best_individs_history:
            individs = self.best_individs_history[iter]
            for individ in individs:
                metric = individ.loss
                if (len(metrics) - 1) < iter:
                    metrics.append(metric)
                elif metrics[-1] > metric:
                    metrics[-1] = metric
        plt.scatter(np.arange(len(metrics)), metrics)

        plt.title('Evolution convergence')
        plt.xlabel('Generation')
        plt.ylabel(ylab)
        plt.show()
