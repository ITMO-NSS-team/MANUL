from copy import deepcopy

import numpy as np
import pickle
from matplotlib import pyplot as plt

from evolution.PopulationEvoOperators import PopulationEvoOperators
from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationStructures import Population
from regularizator.ModuleNN import ModelNN


class Evolution:
    def __init__(self, base_individ: DataStructureGraph,
                 population_size: int,
                 iterations: int,
                 model_to_optimize: ModelNN,
                 base_mutation=True,
                 edges_mutation=True,
                 edges_weight_mutation=True,
                 evo_operators_params: dict = None):
        """
        Class for set and run evolution optimization on graph for best model fitting on data
        :param base_individ: DataStructureGraph class object as starting point
        :param population_size: number of individs in population to produce from base_individ with mutation
        :param iterations: number of iterations for evolution
        :param model_to_optimize: ModelNN for which the best quality is searching
        :param evo_operators_params: dictionary with parameters for evolutionary operators for custom setting
        """
        self.evolution_history = None
        self.base_individ = base_individ
        self.population_size = population_size
        self.iterations = iterations
        self.base_model = model_to_optimize
        self.base_mutation = base_mutation
        self.edges_mutation = edges_mutation
        self.edges_weight_mutation = edges_weight_mutation
        self._init_evo_operators_parameters(evo_operators_params)

        self.population = self.init_population()

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

    def init_population(self):
        population = Population(base_individ=self.base_individ, size=self.population_size).generate(
            base_mutation=self.base_mutation,
            edges_mutation=self.edges_mutation,
            edges_weight_mutation=self.edges_weight_mutation)
        return population

    def evaluate_fitness(self):
        self.population = self.population.evaluate_individs_fitness(self.base_model)

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

    def plot_evolution_pareto_fronts(self, save_path: str = None):
        generations_num = len(self.evolution_history.keys())
        generations_colors = [np.random.rand(3, ) for _ in range(generations_num)]
        for generation in range(generations_num):
            generation_energy = [self.evolution_history[generation][g]['energy'] for g in
                                 range(self.population_size) if self.evolution_history[generation][g]['pareto_best']]
            generation_errors = [self.evolution_history[generation][g]['model_error'] for g in
                                 range(self.population_size) if self.evolution_history[generation][g]['pareto_best']]
            for i in range(len(generation_errors)):
                plt.scatter(generation_errors[i], generation_energy[i], c=generations_colors[generation], alpha=0.3)
                plt.annotate(f'gen: {generation}', (generation_errors[-1], generation_energy[-1]))
        if save_path is not None:
            plt.savefig(f'{save_path}')
            plt.close()
        else:
            plt.show()

    def run(self):
        evolution_history = {}
        self.evaluate_fitness()

        evolution_history[0] = []
        for individ in self.population.individs_pool:
            individ_parameters = {"adjacency_matrix": individ.adjacency_matrix,
                                  "matrix_connect": individ.matrix_connect,
                                  "basis": individ.basis,
                                  "fitness": individ.fitness,
                                  "energy": individ.energy,
                                  "model_error": individ.model_error,
                                  "pareto_best": individ.pareto_best}
            evolution_history[0].append(individ_parameters)

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
                                            base_mutation=self.base_mutation,
                                            edges_mutation=self.edges_mutation,
                                            edges_weight_mutation=self.edges_weight_mutation)

            print('Update fitnesses')
            self.evaluate_fitness()

            print('Filter population')
            #pop_operators.filter_population(self.population_size)
            pop_operators.filter_population_multicriteria(self.population_size)

            evolution_history[i + 1] = []
            for individ in self.population.individs_pool:
                individ_parameters = {"adjacency_matrix": individ.adjacency_matrix,
                                      "matrix_connect": individ.matrix_connect,
                                      "basis": individ.basis,
                                      "fitness": individ.fitness,
                                      "energy": individ.energy,
                                      "model_error": individ.model_error,
                                      "pareto_best": individ.pareto_best}
                evolution_history[i + 1].append(individ_parameters)
                individ.selected = False
                individ.elitism = False

        self.evolution_history = evolution_history


        # overwrite base_individ and base model to best individ
        #TODO overwrite best individ logic



        '''best_individ_index = [ind.fitness for ind in self.population.individs_pool].index(
            max([ind.fitness for ind in self.population.individs_pool]))
        self.base_individ = self.population.individs_pool[best_individ_index]
        self.base_model = self.base_model.train(self.base_individ)
        self.base_individ.save_cache_object(name='final_graph')
        self.save_history(self.evolution_history, name='evolution_history')'''

    def save_history(self, history: dict, name: str = None):
        if name is None:
            name = "history_from_evolution"

        with open(f'{self.base_individ.cache_folder}/{name}.pkl', 'wb') as outp:
            pickle.dump(history, outp, pickle.HIGHEST_PROTOCOL)
            print(f'History evolution saved to {self.base_individ.cache_folder}/{name}.pkl')

    @property
    def pareto(self):
        """
        Property that return pareto front of best graph individs in last generation of evolution
        by two criteria - model loss and energy
        """
        pareto_best = [ind for ind in self.population.individs_pool if ind.pareto_best]
        return pareto_best

    def plot_pareto(self, save_path=None):
        """
        Function to plot pareto front of best individs by two criteria - model loss and energy
        """
        individs_criterias = [[ind.model_error, ind.energy] for ind in self.pareto]
        for i, point in enumerate(individs_criterias):
            plt.scatter(point[0], point[1], c='r')
            plt.annotate(str(i), (point[0], point[1]))
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def pareto_models(self):
        """
        Function for training base model on pareto front individs
        """
        trained_models = []
        for individ in self.pareto:
            individ_model = deepcopy(self.base_model)
            individ_model.train(individ)
            trained_models.append(individ_model)
        return trained_models
