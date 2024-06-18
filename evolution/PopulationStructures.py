from copy import deepcopy

from evolution.PopulationEvoOperators import IndividEvoOperators


class Population:
    def __init__(self, size: int, base_individ):
        """
        Class for generation population from single individ with mutation operator
        :param size: int - number of individs in population
        :param base_individ: DataStructureGraph with individ to mutate
        """
        self.size = size
        self.base_individ = base_individ
        self.individs_pool = []

    def generate(self, base_mutation: bool = True,
                          edges_mutation: bool = True,
                          edges_weight_mutation: bool = True,
                 nodes_mutation_prob: float = None):
        """
        Function to generate population from base individ to pool by mutation
        :param nodes_mutation_prob: percentage of nodes for mutation
        :return: class object with individs pool
        """
        if nodes_mutation_prob is None:
            nodes_mutation_prob = 0.1

        individs_pool = [self.base_individ]
        for i in range(1, self.size):
            print(f'Generate individ {i} / {self.size}')
            new_indvid = deepcopy(self.base_individ)
            mutator = IndividEvoOperators([new_indvid],
                                          base_mutation,
                                          edges_mutation,
                                          edges_weight_mutation)
            new_indvid = mutator.mutate(nodes_mutation_prob=nodes_mutation_prob)[0]
            individs_pool.append(new_indvid)

        self.individs_pool = individs_pool
        return self

    def evaluate_individs_fitness(self, base_model):
        """
        Function for adding fitness parameter to each individ of population
        :param base_model: model in which loss with individ is calculated
        """
        for individ in self.individs_pool:
            if individ.fitness is None:
                individ_model = deepcopy(base_model)
                individ_model.train(individ)
                fitness = individ_model.trained_loss_values['combined_loss']
                fitness = 1 / fitness
                individ.fitness = fitness
        return self

