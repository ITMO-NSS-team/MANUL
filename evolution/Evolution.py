import numpy as np

from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationStructures import Population
from regularizator.ModuleNN import ModelNN


class Evolution:
    def __init__(self, base_individ: DataStructureGraph,
                 population_size: int,
                 iterations: int,
                 model_to_optimize: ModelNN):
        """
        Class for set and run evolution optimization on graph for best model fitting on data
        :param base_individ: DataStructureGraph class object as starting point
        :param population_size: number of individs in population to produce from base_individ with mutation
        :param iterations: number of iterations for evolution
        :param model_to_optimize: ModelNN for which the best quality is searching
        """
        self.base_individ = base_individ
        self.population_size = population_size
        self.iterations = iterations
        self.base_model = model_to_optimize

        self.population = self.init_population()

    def init_population(self):
        population = Population(base_individ=self.base_individ, size=self.population_size).generate()
        return population

    def evaluate_fitness(self):
        self.population = self.population.evaluate_individs_fitness(self.base_model)

    def run(self):
        # TODO праметры для эволюционных операторов вынести
        # TODO  добавить возможность выбора какие операторы применять
        evolution_history = {}
        self.evaluate_fitness()
        evolution_history[0] = [ind.fitness for ind in self.population]
        for i in range(self.iterations):
            pop_operators = PopulationEvoOperators(population=self.population)
            pop_operators.elitism(elits_num=1)
            pop_operators.roulette_wheel_selection(tournament_size=None, winners_size=None)
            pop_operators.crossover_population(crossover_size=None)
            pop_operators.mutate_population(mutation_prob=None)
            # TODO прочекать как меняется популяция после каждого изменения
            # TODO  убедиться что размер популяции неизменен - фильтровать ее
            self.evaluate_fitness()
            evolution_history[i] = [ind.fitness for ind in self.population]




class PopulationEvoOperators:
    def __init__(self, population: Population):
        """
        Class for applying available evolutionary operators to population
        :param population: population to modify with operators
        """
        self.population = population

    def elitism(self, elits_num: int = 1):
        """
        Operator to mark individs with best fitness function
        """
        elite_idxs = np.argsort(list(map(lambda ind: ind.fitness,
                                         self.population.individs_pool)))[-elits_num:]
        for idx in elite_idxs:
            self.population.individs_pool[idx].elitism = True

    def roulette_wheel_selection(self, tournament_size: int = None, winners_size: int = None):
        if tournament_size is None:
            tournament_size = self.population.size
        if winners_size is None:
            winners_size = tournament_size // 2

        selected_individs = list(
            np.random.choice(np.arange(len(self.population.individs_pool)), replace=False, size=tournament_size))
        selected_individs = [self.population.individs_pool[i] for i in selected_individs]
        population_fitnesses = list(map(lambda ind: 1 / (ind.fitness + 0.01), selected_individs))
        fits_sum = np.sum(population_fitnesses)
        probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))

        try:
            # TODO отдебажить ошибки и убрать try
            winners = [selected_individs[i] for i in
                       np.random.choice(np.arange(tournament_size), size=winners_size, p=probabilities,
                                        replace=False)]
        except Exception:
            winners = [selected_individs[i] for i in
                       np.random.choice(np.arange(tournament_size), size=winners_size, replace=False)]

        for individ in winners:
            individ.selected = True

    def crossover_population(self, crossover_size: int = None):
        selected_population = list(filter(lambda individ: individ.selected, self.population.individs_pool))
        if crossover_size is None or crossover_size > len(selected_population) // 2:
            crossover_size = len(selected_population) // 2
        selected_individs = [[selected_population[i], selected_population[j]] for i, j in
                             np.random.choice(np.arange(len(selected_population)), replace=False,
                                              size=(crossover_size, 2))]
        for individ1, individ2 in selected_individs:
            mutator = IndividEvoOperators([individ1, individ2])
            new_individs = mutator.crossover_individs()
            self.population.individs_pool.extend(new_individs)

    def mutate_population(self, mutation_prob: int = None):
        selected_population = list(filter(lambda individ: individ.selected, self.population.individs_pool))
        if mutation_prob is None:
            mutation_prob = 0.3
        number_of_individs_to_mutate = int(len(selected_population) * mutation_prob)

        selected_individs = np.random.choice(selected_population, replace=False, size=number_of_individs_to_mutate)

        mutator = IndividEvoOperators(selected_individs)
        # TODO прокинуть параметры мутации
        mutated_individs = mutator.mutate(nodes_mutation_prob=None)
        self.population.individs_pool.extend(mutated_individs)


class IndividEvoOperators:
    def __init__(self, individs: list[DataStructureGraph]):
        """
        Class for applying available evolutionary operators to individs
        :param individs: list with graph individs for changing
        """
        self.individs = individs

    def mutate(self, nodes_mutation_prob: int = None):
        if nodes_mutation_prob is None:
            # TODO вынести вероятность мутации из захардкоженных во внешний дефолтный словарь
            nodes_mutation_prob = 0.3
        for individ in self.individs:
            individ.elitism = False
            fullness_individ = individ.fullness
            num_nodes = individ.number_of_nodes
            number_of_nodes_to_mutate = int(num_nodes * nodes_mutation_prob)
            eds = individ.matrix_connect
            graph = individ.graph

            methods = np.random.choice(np.arange(2), size=number_of_nodes_to_mutate,
                                       p=[fullness_individ / 100, 1 - (fullness_individ / 100)])
            for method in methods:
                if method:
                    # добавление
                    current_laplassian = individ.laplassian
                    nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                    while current_laplassian[nodes[0]][nodes[1]] != 0:
                        nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                    individ.add_edge(nodes[0], nodes[1])
                    individ.check_vn_part(individ.source_data[individ.basis], nodes[0], nodes[1])
                else:
                    # удаление
                    if individ.number_of_edges == 0:
                        continue
                    probability = []
                    edges = []
                    for key in graph:
                        probability.extend(eds[key, graph[key]])
                        elements = [[key, i] for i in graph[key]]
                        edges.extend(elements)
                    probability = probability / np.sum(probability)
                    edge_index = np.random.choice(np.arange(individ.number_of_edges),
                                                  size=1,
                                                  p=probability.astype(np.float64))[0]
                    edge = edges[edge_index]
                    individ.remove_edge(edge[0], edge[1])
        return self.individs

    def crossover_individs(self):
        if len(self.individs) != 2:
            print(
                f'DEBAG LOG: IndividEvoOperators.crossover_individs - len of individs list is {len(self.individs)} instead 2 -  use crossover on first two elements')
        if len(self.individs) == 1:
            print(
                f'DEBAG LOG: IndividEvoOperators.crossover_individs - len of individs list is 1, return unchanged')
            return self.individs

        individ1 = self.individs[0]
        individ2 = self.individs[1]
        individ1.elitism = False
        individ2.elitism = False

        probability = np.array(
            [abs(len(individ1.graph[i]) - len(individ2.graph[i])) for i in range(individ1.number_of_nodes)])
        probability = probability / probability.sum()
        start_node_index = np.random.choice(np.arange(individ1.number_of_nodes), size=1, p=probability)[0]

        subgraph1 = individ1.graph[start_node_index].copy()
        subgraph2 = individ2.graph[start_node_index].copy()

        individ1.replace_subgraph(start_node_index, subgraph2)
        individ2.replace_subgraph(start_node_index, subgraph1)

        self.individs = [individ1, individ2]

        return self.individs
