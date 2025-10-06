from copy import deepcopy
import os
import pickle

import torch

from evolution.PopulationEvoOperators_new import IndividEvoOperators
from structure_approximation.IntrinsicNN import IntrinsicNN


class Population:
    def __init__(self, size: int, base_individ):
        """
        Class for population generation from a single individ by repeatedly applying mutation operator.
        :param size: int - number of individs in population
        :param base_individ: DataStructureGraph with individ to mutate
        """
        self.size = size
        self.base_individ = base_individ
        self.individs_pool = []

    def generate(self, edges_mutation: bool = True,
                 edges_weight_mutation: bool = True):
        """
        Function to generate population from the base individ to pool by mutation.
        :param nodes_mutation_prob: percentage of nodes for mutation
        :return: class object with individs pool
        """

        individs_pool = [self.base_individ]
        for i in range(1, self.size):
            print(f'Generate individ {i} / {self.size}')
            new_indvid = deepcopy(self.base_individ)
            mutator = IndividEvoOperators([new_indvid],
                                          edges_mutation,
                                          edges_weight_mutation)
            new_indvid = mutator.mutate()[0]
            individs_pool.append(new_indvid)

        self.individs_pool = individs_pool
        return self

    def evaluate_individs_fitness(self):
        """
        Function for adding fitness parameter to each individ of the population.
        """
        for individ in self.individs_pool:
            if individ.fitness is None:
                features = torch.tensor(individ.features)
                targets = torch.tensor(individ.targets)
                dims = individ.dimensionality
                individ_model = IntrinsicNN(features,
                                            targets,
                                            dims,
                                            plot_convergence=False,
                                            epochs=500)

                individ_model.train()
                individ.loss = individ_model.loss
                fitness = 1 / individ.loss
                individ.fitness = fitness

        return self

    def evaluate_individs_criteria(self, base_model):
        # TODO adapt criteria
        """
        Function for adding criteria to each individ of the population
        (implemented for multi-criterion optimization).
        :param base_model: model in which loss with individ is calculated
        """
        for individ in self.individs_pool:
            if individ.fitness is None:
                individ_model = deepcopy(base_model)
                individ_model.train()
                fitness = 1 / individ_model.loss
                individ.fitness = fitness
                individ.criteria = [individ_model.loss, individ.number_of_edges]
                individ.loss = individ_model.loss
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
