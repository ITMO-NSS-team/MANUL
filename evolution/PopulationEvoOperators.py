from copy import deepcopy
from itertools import combinations
import collections
from operator import itemgetter
import matplotlib.pyplot as plt
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
        probabilities = population_fitnesses / fits_sum

        try:
            # TODO debug and remove try
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
        crossover_size = int(len(selected_population) * crossover_size_percent)
        if crossover_size > len(selected_population) // 2:
            crossover_size = len(selected_population) // 2

        selected_individs = [[selected_population[i], selected_population[j]] for i, j in
                             np.random.choice(np.arange(len(selected_population)), replace=False,
                                              size=(crossover_size, 2))]

        for individ1, individ2 in selected_individs:
            # individ1 = deepcopy(individ1)
            # individ2 = deepcopy(individ2)
            mutator = IndividEvoOperators([individ1, individ2])
            new_individs = mutator.crossover_individs()
            self.population.individs_pool.extend(new_individs)

    def mutate_population(self, base_mutation: bool,
                          edges_mutation: bool,
                          edges_weight_mutation: bool,
                          mutation_prob: int = None):
        selected_population = self.population.individs_pool
        if mutation_prob is None:
            mutation_prob = 0.3
        number_of_individs_to_mutate = int(len(selected_population) * mutation_prob)

        selected_individs = np.random.choice(selected_population, replace=False,
                                             size=number_of_individs_to_mutate).tolist()
        # selected_individs = [deepcopy(ind) for ind in selected_individs]

        mutator = IndividEvoOperators(selected_individs, base_mutation, edges_mutation, edges_weight_mutation)
        # TODO throw mutation parameters to upper layers
        mutated_individs = mutator.mutate()
        self.population.individs_pool.extend(mutated_individs)

    def filter_population_multicriteria(self, size_to_save, plot_pareto_selection=True):
        individs_criterias = []
        for i, individ in enumerate(self.population.individs_pool):
            criteria1 = individ.model_error
            criteria2 = individ.energy
            individs_criterias.append([criteria1, criteria2])

        individs_numbers = np.arange(len(individs_criterias))
        cands_to_del = []
        ind_pairs = combinations(individs_numbers, 2)
        for pair in ind_pairs:
            candidate1 = individs_criterias[pair[0]]
            candidate2 = individs_criterias[pair[1]]
            comparison = np.array(candidate1) < np.array(candidate2)
            if np.all(comparison):  # if candidate1 is better
                cands_to_del.append(pair[1])
            if not np.any(comparison):  # if candidate2 is better
                cands_to_del.append(pair[0])

        val, freq = np.unique(cands_to_del, return_counts=True)
        # find individs which are not dominated - best ones
        best_inds = [i for i in individs_numbers if i not in val]
        # find number of not best individs which is needed for population size save
        num_not_best = size_to_save - len(best_inds)
        not_best_inds = []
        if num_not_best > 0:
            not_best_dict = collections.OrderedDict(
                sorted(dict(zip(val, freq)).items(), key=itemgetter(1), reverse=True))
            not_best_inds = list(not_best_dict.keys())[-num_not_best:]

        next_generation = []
        for ind in best_inds:
            self.population.individs_pool[ind].pareto_best = True
            next_generation.append(self.population.individs_pool[ind])
        for ind in not_best_inds:
            next_generation.append(self.population.individs_pool[ind])

        self.population.individs_pool = next_generation

        if plot_pareto_selection:
            # blue points for initial population
            for i, point in enumerate(individs_criterias):
                plt.scatter(point[0], point[1], c='b')
                plt.annotate(str(i), (point[0], point[1]))

            # red points for next generation selected individs
            for i, individ in enumerate(self.population.individs_pool):
                criteria1 = individ.model_error
                criteria2 = individ.energy
                if i < len(best_inds):
                    color = 'r'
                else:
                    color = 'orange'
                plt.scatter(criteria1, criteria2, c=color)

            plt.xlabel('model error')
            plt.ylabel('energy')
            plt.show()

    '''def filter_population(self, size_to_save):
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

        # crop population to fixed size

        self.population.individs_pool = [x for _, x in sorted(zip([fitnesses[i] for i in uniq_inds],
                                                                  self.population.individs_pool),
                                                              key=lambda pair: pair[0])][-size_to_save + len(elite):]

        # if unique individs less than population size extend with elite
        while len(self.population.individs_pool) <= size_to_save - len(elite):
            self.population.individs_pool.extend(elite)'''
