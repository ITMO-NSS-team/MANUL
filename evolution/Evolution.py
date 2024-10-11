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

    def run(self):
        evolution_history = {}
        best_individs_history = {}
        self.evaluate_fitness()

        individ_parameters_dict = {}
        for k, individ in enumerate(self.population.individs_pool):
            individ_parameters_dict[k] = {'fitness': individ.fitness,
                                          'basis': individ.basis}
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
                                            base_mutation=self.base_mutation,
                                            edges_mutation=self.edges_mutation,
                                            edges_weight_mutation=self.edges_weight_mutation)

            print('Update fitnesses')
            self.evaluate_fitness()

            print('Filter population')
            pop_operators.filter_population(self.population_size)
            for individ in self.population.individs_pool:
                if individ.elitism: best_individs_history[i] = {"adjacency_matrix": individ.adjacency_matrix,
                                                                "matrix_connect": individ.matrix_connect, 
                                                                "basis": individ.basis,
                                                                "fitness": individ.fitness,
                                                                "trained_loss_values": individ.trained_loss_values}
                individ.selected = False
                individ.elitism = False

            individ_parameters_dict = {}
            for k, individ in enumerate(self.population.individs_pool):
                individ_parameters_dict[k] = {'fitness': individ.fitness,
                                              'basis': individ.basis}

            evolution_history[i + 1] = individ_parameters_dict
        self.evolution_history = evolution_history

        # overwrite base_individ and base model to best individ
        best_individ_index = [ind.fitness for ind in self.population.individs_pool].index(
            max([ind.fitness for ind in self.population.individs_pool]))
        self.base_individ = self.population.individs_pool[best_individ_index]
        self.base_model = self.base_model.train(self.base_individ)
        self.base_individ.save_cache_object(name='final_graph')
        self.save_history(best_individs_history, name='best_individs_by_iterations')

    def save_history(self, history: dict, name: str = None):
        if name is None:
            name = "history_from_evolution"

        with open(f'{self.base_individ.cache_folder}/{name}.pkl', 'wb') as outp:
            pickle.dump(history, outp, pickle.HIGHEST_PROTOCOL)
            print(f'History evolution saved to {self.base_individ.cache_folder}/{name}.pkl')


class MultiEvolution(Evolution):
    def __init__(self, base_individ: DataStructureGraph,
                 population_size: int,
                 iterations: int,
                 model_to_optimize: ModelNN,
                 base_mutation=True,
                 edges_mutation=True,
                 edges_weight_mutation=True,
                 evo_operators_params: dict = None):
        
        super().__init__(base_individ, population_size, iterations, model_to_optimize, base_mutation, edges_mutation, edges_weight_mutation, evo_operators_params)

        self.__init_weights_vector()
    
    def __init_weights_vector(self):
        x = np.linspace(0, 1, self.population_size)
        y = x[-1::-1]
        self.weights_vector = np.array([x, y]).T

    def plot_vectors(self):
        # origin_point = np.zeros(shape=(2, self.population_size))
        for i in range(self.population_size):
            plt.plot([0,self.weights_vector[i][0]], [0, self.weights_vector[i][1]])
        plt.show()

    def run(self):
        evolution_history = {}
        best_individs_history = {}
        self.evaluate_criteria()

        individ_parameters_dict = {}
        for k, individ in enumerate(self.population.individs_pool):
            individ_parameters_dict[k] = {'fitness': individ.fitness,
                                          'basis': individ.basis}
        evolution_history[0] = individ_parameters_dict

        for i in range(self.iterations):
            print(f'Evolution run, iteration - {i}')
            pop_operators = PopulationEvoOperators(population=self.population)
            self.decomposition_population_by_vectors()
            
            for vector in self.weights_vector:
                print('Search non-dominant individs')
                print('Selecting individs')
                print('Crossover')
                print('Mutate')
                print('Count criterias in new individs')
                print('Replace individs')

    def evaluate_criteria(self):
        self.population = self.population.evaluate_individs_criteria(self.base_model)

    def subsidiary_method(self, vector, criteria):
        norm_of_vector = np.linalg.norm(vector)
        d1 = np.linalg.norm(criteria - np.array([0, 0]).T * vector) / norm_of_vector
        d2 = np.linalg.norm(criteria - (np.array([0, 0]) + d1 *(vector/np.linalg.norm(vector))))
        lmd = 3

        return d1 + lmd * d2

    def decomposition_population_by_vectors(self):
        new_structure = []
        current_structure = self.population.individs_pool
        for vector in self.weights_vector:
            result = map(lambda ind: self.subsidiary_method(vector, ind.criteria), current_structure)
            result = np.argmin(result)
            new_structure.append(current_structure[result])
            current_structure = current_structure[0:result] + current_structure[result+1:]
        
        self.population.individs_pool = new_structure