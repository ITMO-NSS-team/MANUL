import numpy as np

from Eva.IndividEvoOperators_new import IndividEvoOperators
from Eva.PopulationStructures_new import Population


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

    def rank_based_selection(self, tournament_size: int = None, winners_size: int = None):
        if tournament_size is None:
            tournament_size = self.population.size
        if winners_size is None:
            winners_size = tournament_size // 2

        # Select random individuals for tournament
        selected_individs = list(
            np.random.choice(np.arange(len(self.population.individs_pool)), replace=False, size=tournament_size))
        selected_individs = [self.population.individs_pool[i] for i in selected_individs]

        # Sort by fitness (descending - higher fitness is better)
        sorted_individs = sorted(selected_individs, key=lambda ind: ind.fitness, reverse=True)

        # Assign ranks (rank 1 is best, rank N is worst)
        ranks = np.arange(len(sorted_individs), 0, -1)  # [N, N-1, ..., 1]

        # Calculate probabilities based on ranks
        # You can use linear or exponential ranking
        probabilities = ranks / np.sum(ranks)  # Linear ranking

        # Select winners based on rank probabilities
        winners_indices = np.random.choice(
            np.arange(len(sorted_individs)),
            size=winners_size,
            p=probabilities,
            replace=False
        )

        winners = [sorted_individs[i] for i in winners_indices]

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
            mutator = IndividEvoOperators([individ1, individ2])
            new_individs = mutator.crossover_individs()
            self.population.individs_pool.extend(new_individs)

    def mutate_population(self,
                          edges_mutation: bool,
                          edges_weight_mutation: bool,
                          mutation_prob: int = None):
        selected_population = self.population.individs_pool
        if mutation_prob is None:
            mutation_prob = 0.3
        number_of_individs_to_mutate = int(len(selected_population) * mutation_prob)

        selected_individs = np.random.choice(selected_population, replace=False,
                                             size=number_of_individs_to_mutate).tolist()

        mutator = IndividEvoOperators(selected_individs, edges_mutation, edges_weight_mutation)
        mutated_individs = mutator.mutate()
        self.population.individs_pool.extend(mutated_individs)

    def filter_population(self, size_to_save):
        # Filter by eigenvalues
        '''valid_indices = [i for i, ind in enumerate(self.population.individs_pool)
                         if not ind.valid_eigenvalues]
        for i in sorted(valid_indices, reverse=True):
            del self.population.individs_pool[i]'''

        # Filter elite individuals
        elite_indices = [i for i, ind in enumerate(self.population.individs_pool)
                         if ind.elitism]
        elite = [self.population.individs_pool[i] for i in elite_indices]
        for i in sorted(elite_indices, reverse=True):
            del self.population.individs_pool[i]

        # Filter duplicates
        fitnesses = [f.fitness for f in self.population.individs_pool]
        uniq_fitnesses = set(fitnesses)
        uniq_indices = [i for i, e in enumerate(fitnesses) if e in uniq_fitnesses]
        self.population.individs_pool = [self.population.individs_pool[i] for i in uniq_indices]

        # crop population to fixed size
        self.population.individs_pool = [x for _, x in sorted(zip([fitnesses[i] for i in uniq_indices],
                                                                  self.population.individs_pool),
                                                              key=lambda pair: pair[0])][-size_to_save + len(elite):]

        # if unique individs less than population size extend with elite
        while len(self.population.individs_pool) <= size_to_save - len(elite):
            self.population.individs_pool.extend(elite)
