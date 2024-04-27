from copy import deepcopy

import numpy as np

from evolution.IndividEvoOperators import IndividEvoOperators
from evolution.PopulationStructures import Population


class PopulationEvoOperators:
    def __init__(self, population: Population):
        """
        Class for applying available evolutionary operators to population
        :param population: population to modify with operators
        """
        self.population = population

    def elitism(self, elits_num: int = None):
        """
        Operator to mark individs with the best fitness function
        :param elits_num: number of best individs to mark as elite
        """
        if elits_num is None:
            elits_num = 1
        elite_idxs = np.argsort(list(map(lambda ind: ind.fitness,
                                         self.population.individs_pool)))[-elits_num:]
        for idx in elite_idxs:
            self.population.individs_pool[idx].elitism = True
            self.population.individs_pool[idx].selected = True

    def roulette_wheel_selection(self, tournament_size: int = None, winners_size: int = None):
        if tournament_size is None:
            tournament_size = self.population.size
        if winners_size is None:
            winners_size = tournament_size // 2

        selected_individs = list(
            np.random.choice(np.arange(len(self.population.individs_pool)), replace=False, size=tournament_size))
        selected_individs = [self.population.individs_pool[i] for i in selected_individs]
        population_fitnesses = [ind.fitness for ind in selected_individs]
        fits_sum = np.sum(population_fitnesses)
        probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))

        try:
            # TODO отдебажить ошибки и убрать try
            winners = [selected_individs[i] for i in
                       np.random.choice(np.arange(tournament_size), size=winners_size, p=probabilities,
                                        replace=False)]
        except Exception as e:
            winners = [selected_individs[i] for i in
                       np.random.choice(np.arange(tournament_size), size=winners_size, replace=False)]

        for individ in winners:
            individ.selected = True

    def crossover_population(self, crossover_size_percent: int = None):
        selected_population = list(filter(lambda individ: individ.selected, self.population.individs_pool))
        if crossover_size_percent is None:
            crossover_size_percent = 0.5
        crossover_size = int(len(selected_population)*crossover_size_percent)
        if crossover_size > len(selected_population) // 2:
            crossover_size = len(selected_population) // 2
        selected_individs = [[selected_population[i], selected_population[j]] for i, j in
                             np.random.choice(np.arange(len(selected_population)), replace=False,
                                              size=(crossover_size, 2))]

        for individ1, individ2 in selected_individs:
            individ1 = deepcopy(individ1)
            individ2 = deepcopy(individ1)
            mutator = IndividEvoOperators([individ1, individ2])
            new_individs = mutator.crossover_individs()
            self.population.individs_pool.extend(new_individs)

    def mutate_population(self, mutation_prob: int = None):
        selected_population = list(filter(lambda individ: individ.selected, self.population.individs_pool))
        if mutation_prob is None:
            mutation_prob = 0.3
        number_of_individs_to_mutate = int(len(selected_population) * mutation_prob)

        selected_individs = np.random.choice(selected_population, replace=False, size=number_of_individs_to_mutate).tolist()
        selected_individs = [deepcopy(ind) for ind in selected_individs]

        mutator = IndividEvoOperators(selected_individs)
        # TODO прокинуть параметры мутации
        mutated_individs = mutator.mutate(nodes_mutation_prob=None)
        self.population.individs_pool.extend(mutated_individs)


    def fiter_population(self, size_to_save):
        elite_inds = [ind.elitism for ind in self.population.individs_pool]
        elite_inds = [i for i, x in enumerate(elite_inds) if x]
        elite = [self.population.individs_pool[i] for i in elite_inds]
        for i in elite_inds:
            self.population.individs_pool.remove(self.population.individs_pool[i])
        fitnesses = [f.fitness for f in self.population.individs_pool]
        # filter individ duplicates
        uniq_fitnesses = set(fitnesses)
        uniq_inds = [i for i, e in enumerate(fitnesses) if e in uniq_fitnesses]
        self.population.individs_pool = [self.population.individs_pool[i] for i in uniq_inds]
        self.population.individs_pool = [x for _, x in sorted(zip([fitnesses[i] for i in uniq_inds], self.population.individs_pool),
                                                              key=lambda pair: pair[0])][-size_to_save + len(elite):]

        # if unique individs less than population size extend with elite
        while len(self.population.individs_pool) <= size_to_save - len(elite):
            self.population.individs_pool.extend(elite)






