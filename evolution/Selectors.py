import numpy as np


class Elitism:
    def __init__(self, params: dict = None):
        """

        :param params:
        :type params:
        """

    def apply(self, population, *args, **kwargs):
        for individ in population.structure:
            individ.elitism = False
        elite_idxs = np.argsort(list(map(lambda ind: ind.fitness,
                                         population.structure)))[-self.params['elitism']:]

        for idx in elite_idxs:
            population.structure[idx].elitism = True
            population.base_model = population.structure[idx].model.copy()
            population.change_fitness.append(population.structure[idx].fitness)


class RouletteWheelSelection:
    """
    """

    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs) -> None:
        pops = population.structure
        assert len(pops) > 0, 'Empty population'
        for individ in pops:
            individ.selected = False

        tournament_size = self.params['tournament_size']
        winners_size = self.params['winners_size']

        if tournament_size is None or tournament_size > len(pops):
            tournament_size = len(pops)
            winners_size = round(tournament_size / 100 * 50)

        selected_individs = list(np.random.choice(np.arange(len(pops)), replace=False, size=tournament_size))
        selected_individs = [pops[i] for i in selected_individs]
        population_fitnesses = list(map(lambda ind: 1 / (ind.fitness + 0.01), selected_individs))
        fits_sum = np.sum(population_fitnesses)
        probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))

        if winners_size is None or winners_size == 0:
            winners = selected_individs
        else:
            try:
                winners = [selected_individs[i] for i in
                           np.random.choice(np.arange(tournament_size), size=winners_size, p=probabilities,
                                            replace=False)]
            except:
                winners = [selected_individs[i] for i in
                           np.random.choice(np.arange(tournament_size), size=winners_size, replace=False)]

        for individ in winners:
            individ.selected = True


class FilterPopulation:
    def __init__(self, params: dict = None):
        super().__init__(params)

    def apply(self, population, *args, **kwargs):
        new_structure = []
        for individ in population.structure:
            if individ not in new_structure:
                new_structure.append(individ)

        if self.params['population_size'] < len(new_structure):
            elite = list(filter(lambda ind: ind.elitism, new_structure))

            for individ in elite:
                new_structure.remove(individ)

            population_fitnesses = list(map(lambda ind: ind.fitness + 0.01, new_structure))
            fits_sum = np.sum(population_fitnesses)
            probabilities = list(map(lambda x: x / fits_sum, population_fitnesses))
            # print(self.params['population_size']-1, len(probabilities))
            new_structure = list(np.random.choice(new_structure, size=self.params['population_size'] - 1,
                                                  p=probabilities, replace=False))
            new_structure.extend(elite)

        population.structure = new_structure

        # for individ in population.structure:
        #    individ.check_visible(self.params["source_data"])
        #     print("CHEcK done")