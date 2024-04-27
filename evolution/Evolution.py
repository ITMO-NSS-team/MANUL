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
        population = Population(base_individ=self.base_individ, size=self.population_size).generate()
        return population

    def evaluate_fitness(self):
        self.population = self.population.evaluate_individs_fitness(self.base_model)

    def plot_evolution_fitnesses(self):
        for generation in range(len(self.evolution_history.keys())):
            generation_fitnesses = [self.evolution_history[generation][g]['fitness'] for g in
                                    self.evolution_history[generation].keys()]
            plt.scatter([generation] * len(self.evolution_history[generation]), generation_fitnesses)

        plt.title('Evolution convergence')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def run(self):
        evolution_history = {}
        self.evaluate_fitness()

        individ_parameters_dict = {}
        for k, individ in enumerate(self.population.individs_pool):
            individ_parameters_dict[k] = {'fitness': individ.fitness,
                                          'basis': individ.basis,
                                          'graph': individ.graph}
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
                                            get('mutation_prob', None))

            print('Update fitnesses')
            self.evaluate_fitness()

            print('Filter population')
            pop_operators.fiter_population()

            individ_parameters_dict = {}
            for k, individ in enumerate(self.population.individs_pool):
                individ_parameters_dict[k] = {'fitness': individ.fitness,
                                              'basis': individ.basis,
                                              'graph': individ.graph}

            evolution_history[i + 1] = individ_parameters_dict
        self.evolution_history = evolution_history

        # overwrite base_individ and base model to best individ
        best_individ_index = [ind.fitness for ind in self.population.individs_pool].index(max([ind.fitness for ind in self.population.individs_pool]))
        self.base_individ = self.population.individs_pool[best_individ_index]
        self.base_model = self.population.individs_models[best_individ_index]


