import math
from copy import deepcopy

import numpy as np

from evolution.IndividStructures import DataStructureGraph


class IndividEvoOperators:
    def __init__(self, individs: list[DataStructureGraph]):
        """
        Class for applying available evolutionary operators to individs
        :param individs: list with graph individs for changing
        """
        self.individs = individs

    def mutate(self, nodes_mutation_prob: float = None):
        if nodes_mutation_prob is None:
            nodes_mutation_prob = 0.5
        if nodes_mutation_prob >= 1:
            raise Exception(
                f'IndividEvoOperators.mutate nodes_mutation_prob={nodes_mutation_prob} should be from 0 to 1')
        append = False
        for individ in self.individs:
            if individ.elitism:
                individ = deepcopy(individ)
                individ.fitness = False
                append = True  # create new individ of modify existing
            individ.elitism = False
            fullness_individ = individ.fullness
            num_nodes = individ.number_of_nodes
            number_of_nodes_to_mutate = int(num_nodes * nodes_mutation_prob)
            eds = individ.matrix_connect
            graph = individ.graph

            nodes_mutation_flags = np.random.choice(np.arange(2), size=number_of_nodes_to_mutate)
            for flag in nodes_mutation_flags:
                if flag:
                    mean_dist_from_neighbours = []
                    for node in graph:
                        dist_from_neighbours = []
                        for neighbour in graph[node]:
                            dist_from_neighbours.append(eds[node, neighbour])
                        if len(dist_from_neighbours) == 0:
                            mean_dist_from_neighbours.append(np.nan)
                        else:
                            mean_dist_from_neighbours.append(np.nanmean(np.array(dist_from_neighbours)))

                    mean_dist = float(np.nanmean(mean_dist_from_neighbours))
                    mean_dist_from_neighbours = np.nan_to_num(mean_dist_from_neighbours, nan=mean_dist)
                    mean_dist_from_neighbours = max(mean_dist_from_neighbours) - mean_dist_from_neighbours
                    probability = mean_dist_from_neighbours / np.sum(mean_dist_from_neighbours)
                    # чем ближе узел к соседям, тем ниже вероятность его выбрать
                    node_index = np.random.choice(np.arange(individ.number_of_nodes),
                                                  size=1,
                                                  p=probability.astype(np.float64))[0]
                    individ.twist_node(node_index)

            edges_mutation_flags = np.random.choice(np.arange(2), size=number_of_nodes_to_mutate,
                                                    p=[fullness_individ / 100, 1 - (fullness_individ / 100)])
            for flag in edges_mutation_flags:
                if flag:
                    # добавление ребра
                    current_laplassian = individ.laplassian
                    nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                    while current_laplassian[nodes[0]][nodes[1]] != 0:
                        nodes = np.random.choice(np.arange(num_nodes), size=2, replace=False)
                    individ.add_edge(nodes[0], nodes[1])
                    individ.check_vn_part(individ.source_data[individ.basis], nodes[0], nodes[1])
                else:
                    # удаление ребра
                    if individ.number_of_edges == 0:
                        continue
                    probability = []
                    edges = []
                    for node in graph:
                        probability.extend(eds[node, graph[node]])  # типа длина каждого ребра с соседями

                        elements = [[node, i] for i in graph[node]]
                        edges.extend(elements)
                    probability = probability / np.sum(probability)
                    edge_index = np.random.choice(np.arange(individ.number_of_edges),
                                                  size=1,
                                                  p=probability.astype(np.float64))[0]
                    edge = edges[edge_index]
                    individ.remove_edge(edge[0], edge[1])
            if append:
                self.individs.append(individ)
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

        if probability.sum() == 0:
            start_node_index = np.random.choice(np.arange(individ1.number_of_nodes), size=1)[0]
        else:
            probability = probability / probability.sum()
            start_node_index = np.random.choice(np.arange(individ1.number_of_nodes), size=1, p=probability)[0]

        subgraph1 = individ1.graph[start_node_index].copy()
        subgraph2 = individ2.graph[start_node_index].copy()

        individ1.replace_subgraph(start_node_index, subgraph2)
        individ2.replace_subgraph(start_node_index, subgraph1)

        self.individs = [individ1, individ2]

        return self.individs
