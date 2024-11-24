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
        crossover_size = int(len(selected_population)*crossover_size_percent)
        if crossover_size > len(selected_population) // 2:
            crossover_size = len(selected_population) // 2

        selected_individs = [[selected_population[i], selected_population[j]] for i, j in
                             np.random.choice(np.arange(len(selected_population)), replace=False,
                                              size=(crossover_size, 2))]

        for individ1, individ2 in selected_individs:
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

        selected_individs = np.random.choice(selected_population, replace=False, size=number_of_individs_to_mutate).tolist()

        mutator = IndividEvoOperators(selected_individs, base_mutation, edges_mutation, edges_weight_mutation)
        # TODO throw mutation parameters to upper layers
        mutated_individs = mutator.mutate()
        self.population.individs_pool.extend(mutated_individs)


    def filter_population(self, size_to_save):
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

        # test code - if individs are chosen by probability
        '''fitnesses = [f.fitness for f in self.population.individs_pool]
        fitnesses = (fitnesses - np.min(fitnesses)) / (np.max(fitnesses) - np.min(fitnesses))
        posibs_to_live = fitnesses / np.sum(fitnesses)
        alive_inds_indices = np.random.choice(len(fitnesses), size=size_to_save-len(elite), p=posibs_to_live, replace=False)
        self.population.individs_pool = [self.population.individs_pool[i] for i in alive_inds_indices]'''

        self.population.individs_pool = [x for _, x in sorted(zip([fitnesses[i] for i in uniq_inds],
                                                                  self.population.individs_pool),
                                                              key=lambda pair: pair[0])][-size_to_save + len(elite):]

        # if unique individs less than population size extend with elite
        while len(self.population.individs_pool) <= size_to_save - len(elite):
            self.population.individs_pool.extend(elite)


class PopulationMultiEvoOperators(PopulationEvoOperators):

    def __init__(self, population):
        super().__init__(population)

    def scalar_product(self, vector:np.ndarray, criteria: np.ndarray):
        """
        Method for computing angle between base vector and individ criteria
        :param vector: coordinates of base vector
        :param critetia: individ's criteria
        """
        norm_of_vector = np.linalg.norm(vector)
        d1 = np.linalg.norm(criteria - np.array([0, 0]).T * vector) / norm_of_vector
        d2 = np.linalg.norm(criteria - (np.array([0, 0]) + d1 *(vector/np.linalg.norm(vector))))
        lmd = 3

        return d1 + lmd * d2

    def decomposition_population_by_vectors(self, weights_vector: list):
        """
        Method for distributing individuals by base vectors with help of sorting individuals according to the index of the vector 
        :param weights_vector:  the list with vectors
        """
        new_structure = []
        current_structure = self.population.individs_pool
        for vector in weights_vector:
            result = list(map(lambda ind: self.scalar_product(vector, ind.criteria), current_structure))
            result = np.argmin(result)
            new_structure.append(current_structure[result])
            current_structure = current_structure[0:result] + current_structure[result+1:]
        
        self.population.individs_pool = new_structure

    def fast_non_dominated_sorting(self) -> list:
        """
        Procedure of separating points from the general population into non-dominated levels.
        This function is a faster alternative to the ``slow_non_dominated_sorting``, but requires 
        a little more memory to store indexes of elements, dominated by every solution. This 
        method was introduced in *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast 
        and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
        vol. 6, no. 2, pp. 182–197, Apr. 2002.* The computational complexity of the method is 
        :math:`O(MN^2)`, where *N* is the population size, and *M* is the number of objective 
        functions in comparisson with :math:`O(MN^3)` of the straightforward way.

        """
        population = self.population.individs_pool
        # Число элементов, доминирующих над i-ым кандидиатом
        domination_count = np.zeros(len(population))
        # Индексы элементов, над которыми доминирует i-ый кандидиат
        dominated_solutions = [[] for elem_idx in np.arange(len(population))]
        current_level_idxs = []
        for main_elem_idx in np.arange(len(population)):
            for compared_elem_idx in np.arange(len(population)):
                if main_elem_idx == compared_elem_idx:
                    continue
                if self.check_dominance(population[compared_elem_idx], population[main_elem_idx]):
                    domination_count[main_elem_idx] += 1
                elif self.check_dominance(population[main_elem_idx], population[compared_elem_idx]):
                    dominated_solutions[main_elem_idx].append(compared_elem_idx)
            if domination_count[main_elem_idx] == 0:
                current_level_idxs.append(main_elem_idx)

        level_idx = 0
        while len(current_level_idxs) > 0:
            for individ_idx in current_level_idxs: population[individ_idx].level = level_idx
            new_level_idxs = []
            for main_elem_idx in current_level_idxs:
                for dominated_elem_idx in dominated_solutions[main_elem_idx]:
                    domination_count[dominated_elem_idx] -= 1
                    if domination_count[dominated_elem_idx] == 0:
                        new_level_idxs.append(dominated_elem_idx)
            level_idx += 1
            current_level_idxs = new_level_idxs  # deepcopy(new_level_idxs)

        self.population.individs_pool = population
    
    def check_dominance(self, target, compared_with) -> bool:
        """
        Method to check, if one solution is dominated by another.
        :param target: individual solution on the pareto levels, compared with the other element.
        :param compared_with: individual solution on the pareto levels, with with the target is compared.
        :return: method returns True, if the **compared_with** dominates (has at least one objective
                functions with less values, while the others are the same) the **target**; 
                False in all other cases.
        """
        flag = False

        for critea_idx in range(len(target.criteria)):
            if target.criteria[critea_idx] > compared_with.criteria[critea_idx]:
                return False
            if target.criteria[critea_idx] < compared_with.criteria[critea_idx]:
                flag = True
        return flag
    
    def selection_for_multiopt(self, index_vector: int, size=1):
        """
        Selection of individuals for evolution operators (mutation, crossover). The choice depends on current base vector. 
        Probabilities are counted by position of individuals among each other.
        :param index_vector: index current base vector, coincides with the index of the individual corresponding to the vector
        :param size: amount of selected individuals except current vector's individual
        """
        probabilties = []
        other_individs = []
        for ind_idx, individ in enumerate(self.population.individs_pool):
            if ind_idx == index_vector:
                continue
            probabilties.append(abs(ind_idx - index_vector))
            other_individs.append(ind_idx)

        probabilties = probabilties / np.sum(probabilties)

        if size > len(other_individs):
            size = len(other_individs)

        selected_individs = list(
            np.random.choice(other_individs, replace=False, size=size, p=probabilties))
        
        for ind in selected_individs:
            self.population.individs_pool[ind].selected = True
        
        self.population.individs_pool[index_vector].selected = True


    def form_popualtion_with_new_individs(self):
        """
        Replace individuals with low level of dominance to new individuals that were obtained from evolutionary operators.
        """
        new_individs = []
        sort_ind = []
        population = self.population.individs_pool
        i = 0
        while i < len(population):
            individ = population[i]
            if individ.level is None:
                population.pop(i)
                new_individs.append(individ)
                continue
            sort_ind.append(individ.level)
            i += 1

        sort_ind = np.argsort(sort_ind)[::-1]
        for i, individ in enumerate(new_individs):
            index = sort_ind[i]
            self.population.individs_pool[index] = individ