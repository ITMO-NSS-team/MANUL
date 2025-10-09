import math

import numpy as np
from copy import deepcopy

from Eva.IndividStructures_new import DataStructureGraph


class IndividEvoOperators:
    def __init__(self, individs: list[DataStructureGraph],
                 edges_mutation: bool = True,
                 edges_weight_mutation: bool = True,
                 ):
        """
        Class for applying available evolutionary operators to individs
        :param individs: list with graph individs for changing
        """
        self.individs = [deepcopy(ind) for ind in individs]
        for individ in self.individs:
            individ.elitism = False
            individ.fitness = None
            individ.level = None

        self.edges_mutation = edges_mutation
        self.edges_weight_mutation = edges_weight_mutation

    def mutate(self,
               edges_len_mutation_prob: float = 0.4,
               edges_existence_mutation_prob: float = 0.4,
               ):

        for individ in self.individs:
            num_nodes = individ.number_of_nodes
            number_of_edges_to_mutate = int(math.ceil(individ.distances_matrix.size * edges_existence_mutation_prob))

            # GRAPH EDGES MUTATION
            if self.edges_mutation:
                nodes_indices_to_change_edge = np.random.randint(num_nodes, size=(2, number_of_edges_to_mutate))
                # remove circular edges
                nodes_indices_to_change_edge = nodes_indices_to_change_edge[:, nodes_indices_to_change_edge[0] != nodes_indices_to_change_edge[1]]
                edges_values = individ.distances_matrix[nodes_indices_to_change_edge[0], nodes_indices_to_change_edge[1]]

                inds_to_add_edge = nodes_indices_to_change_edge[:, edges_values == 0]
                inds_to_remove_edge = nodes_indices_to_change_edge[:, edges_values != 0]

                '''# fixing number of edges to add and to remove to close values
                min_num = np.min([inds_to_remove_edge.shape[1], inds_to_add_edge.shape[1]])


                if inds_to_add_edge.shape[1] != min_num:
                    inds_to_add_edge = inds_to_add_edge[:, :min_num]
                if inds_to_remove_edge.shape[1] != min_num:
                    inds_to_remove_edge = inds_to_remove_edge[:, :min_num]'''

                individ.add_edges(inds_to_add_edge)
                individ.remove_edges(inds_to_remove_edge)

            # mutate edges length
            if self.edges_weight_mutation:
                num_of_edges_to_mutate = int(individ.number_of_edges * edges_len_mutation_prob)
                mask_matrix = np.tril(np.full(individ.distances_matrix.shape, 1), -1)
                one_way_adj_matrix = mask_matrix * individ.distances_matrix != 0
                edges = np.array(np.where(one_way_adj_matrix == 1))

                if edges.shape[1] > 0:
                    edges_to_mutate_indices = edges[:, np.random.choice(edges.shape[1], size=num_of_edges_to_mutate, replace=False)]
                    individ.change_edges_length(edges_to_mutate_indices, mutate_intensity=0.2)

            for individ in self.individs:
                individ.upd_features()

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

        selected_node_index = np.random.randint(individ1.number_of_nodes)

        subgraph1 = np.array(np.where(individ1.distances_matrix[selected_node_index] != 0))
        subgraph2 = np.array(np.where(individ2.distances_matrix[selected_node_index] != 0))

        individ1.replace_subgraph(selected_node_index, subgraph2)
        individ2.replace_subgraph(selected_node_index, subgraph1)

        self.individs = [individ1, individ2]

        for individ in self.individs:
            individ.upd_features()

        return self.individs
