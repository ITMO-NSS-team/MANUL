from copy import deepcopy
import os
import pickle

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
                individ.trained_loss_values = individ_model.trained_loss_values
        return self
    
    def evaluate_individs_criteria(self, base_model):
        """
        Function for adding criterions to each individ of population
        (for multi-criterion optimization)
        :param base_model: model in which loss with individ is calculated
        """
        for individ in self.individs_pool:
            if individ.fitness is None:
                individ_model = deepcopy(base_model)
                individ_model.train(individ)
                fitness = individ_model.trained_loss_values['combined_loss']
                fitness = 1 / fitness
                individ.fitness = fitness
                individ.criteria = [individ_model.trained_loss_values['combined_loss'], 1/individ_model.trained_loss_values['graph_loss']]
                # individ.criteria = [individ_model.trained_loss_values['model_loss'], individ_model.trained_loss_values['graph_loss']]
                individ.trained_loss_values = individ_model.trained_loss_values
        return self
    
    def load_individs_pool(self, path):
        """
        Function to load self object from pickle file
        :param path: name of file with graph objects .pkl to load in cache folder if individ or absolute path
        """
        if os.path.isfile(path):
            with open(path, 'rb') as inp:
                tmp_dict = pickle.load(inp)
        elif os.path.isfile(f'{self.base_individ.cache_folder}/{path}'):
            with open(f'{self.base_individ.cache_folder}/{path}', 'rb') as inp:
                tmp_dict = pickle.load(inp)
        else:
            raise Exception(f'Failed to load graph object, no such file {path}')

        for key in tmp_dict:
            new_individ = deepcopy(self.base_individ)
            properties = tmp_dict[key]
            new_individ.__dict__.update(properties)
            self.individs_pool.append(new_individ)

