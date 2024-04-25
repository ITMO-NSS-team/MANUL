import ast
import os
import pickle
import numpy as np
import numba.types as tp
from numba import njit
from datetime import datetime

import topo as tp
from sklearn.metrics.pairwise import euclidean_distances


def get_basis(graph, source_data: np.ndarray):
    """
    Method for reducing nodes
    :param source_data: matrix n * m, where n - number of nodes, m - number of features. Keeping values of nodes by fields.
    :return basis : list -  indexes of the nodes that we save in the graph
    """
    named_graph = [{'index': i, 'neighbours': [], 'stamp': False} for i in graph]
    for i in graph:
        for k in graph[i]:
            named_graph[k]['neighbours'].append(i)
            named_graph[i]['neighbours'].append(k)
    basis = []
    temp_graph = list(filter(lambda elem: not elem['stamp'], named_graph))
    while len(temp_graph) > 0:
        max_index = np.argmax([len(elem['neighbours']) for elem in temp_graph])
        use_index = temp_graph[max_index]['neighbours']
        use_index.append(temp_graph[max_index]['index'])
        average_values = np.average(source_data[use_index], axis=0)
        choose_point = use_index[0]
        for indx in use_index:
            if np.sqrt(((source_data[indx] - average_values) ** 2).sum()) < np.sqrt(
                    ((source_data[choose_point] - average_values) ** 2).sum()):
                choose_point = indx
            named_graph[indx]['stamp'] = True
        basis.append(choose_point)
        temp_graph = list(filter(lambda elem: elem['index'] not in use_index, temp_graph))
    return basis


class DataStructureGraph:
    def __init__(self, data: np.ndarray, n_neighbors: int = None, eps: float = None, graph_file: str = None,
                 cash_folder: str = None):
        """
        Class for initialization individ  for evolution as complex graph structure with graph properties
        :param data: features table for graph structure creation
        :param n_neighbors:  number of neighbors to save for node
        :param eps: epsilon distance between neighbors to decrease the closest
        :param graph_file: str - path to file .pkl with DataStructureGraph object
        :param cash_folder: str - path to save cash
        """
        self.elitism = False  # TODO вынести в класс предок - индивида (исп-ся только в эволюции)
        self.selected = False  # TODO вынести в класс предок - индивида (исп-ся только в эволюции)
        self.fitness = None
        self.source_data = data.astype(float)
        if cash_folder is None:
            self.cash_folder = f"info_log/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        else:
            self.cash_folder = cash_folder

        if not os.path.exists(self.cash_folder):
            os.makedirs(self.cash_folder)
        print(f'Log folder set as {self.cash_folder}')

        if eps is None:
            self.epsilon_neighborhood = 0.15
        else:
            self.epsilon_neighborhood = eps
        if n_neighbors is None:
            self.n_neighbors = 10
        else:
            self.n_neighbors = n_neighbors

        if graph_file is not None:
            self.load_cash_object(graph_file)
        else:
            self.find_edges(data, use_kernel=True)
            #  индексы для разреживания графа
            self.basis = get_basis(self.graph, data)
            self.number_of_nodes = len(self.basis)
            # обновление ребер для разреженного графа
            self.find_edges(data[self.basis], use_kernel=False)
            self.filter_graph(data[self.basis])
            self.calc_fullness()
            # сохраняем в кэш
            self.save_cash_object('base_graph')

    def save_cash_object(self, name: str = None):
        """
        Function to save  self object as pickle file
        :param name: string with name without .pkl to save in cash folder
        """
        if name is None:
            name = 'graph_obj'
        with open(f'{self.cash_folder}/{name}.pkl', 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)
            print(f'Graph object saved to {self.cash_folder}/{name}.pkl')

    def load_cash_object(self, name):
        """
        Function to load self object from pickle file
        :param name: name of file with graph object .pkl to load in cash folder
        """
        with open(f'{self.cash_folder}/{name}', 'rb') as inp:
            tmp_dict = pickle.load(inp)
            self.__dict__.update(tmp_dict)

    def loss_function(self, f_x: np.ndarray, indices=None):
        """
        Function for calculation graph loss with f(x) values
        :param f_x: np.ndarray - values of f(x) function for loss calculation
        :param indices: list with batch indices
        :return: float - value of loss function
        """
        laplassian = self.laplassian
        if indices is not None:
            laplassian = laplassian[indices][:, indices]
        part_1 = np.dot(f_x.T, laplassian)
        loss = np.dot(part_1, f_x)
        return loss.reshape(-1)[0]

    def form_graph_with_laplassian(self, distances, laplassian):
        graph = {}
        number_of_edges = 0
        adjacency_matrix = np.zeros(distances.shape)
        for i in range(len(distances)):
            print(f'Process node {i}/{len(distances)}')
            graph[i] = []
            for j in range(i, len(distances)):
                if i == j:
                    continue
                if laplassian[i, j] != 0:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    graph[i].append(j)
                    number_of_edges += 1
        return graph, adjacency_matrix, number_of_edges

    def form_graph_with_euclidian_distances(self, distances, nodes_data):
        graph = {}
        number_of_edges = 0
        adjacency_matrix = np.zeros(distances.shape)
        different = np.zeros((distances.shape[0], distances.shape[0], nodes_data.shape[1]))
        for i in range(len(distances)):
            print(f'Process node {i}/{len(distances)}')
            graph[i] = []
            different[i] = -1 * different[:, i]
            for j in range(i, len(distances)):
                different[i][j] = nodes_data[i] - nodes_data[j]
                if i == j:
                    continue
                if distances[i][j] / np.max(distances) <= self.epsilon_neighborhood:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
                    graph[i].append(j)
                    number_of_edges += 1
            distances = np.array(distances)
        return graph, adjacency_matrix, number_of_edges

    def find_edges(self, nodes_data, use_kernel=True):
        euclid_dists = euclidean_distances(nodes_data, nodes_data)
        matrix_connect = euclid_dists / np.max(euclid_dists)

        if use_kernel:
            print('Form graph (laplassian method)')
            kernel = tp.tpgraph.Kernel(n_neighbors=self.n_neighbors, n_jobs=1, metric='cosine', fuzzy=True,
                                       verbose=True)
            kernel.fit(nodes_data)
            lapl = kernel.L.todense()
            graph, adjacency_matrix, number_of_edges = self.form_graph_with_laplassian(euclid_dists, lapl)

        else:
            print('Form graph (euclidian distances)')
            graph, adjacency_matrix, number_of_edges = self.form_graph_with_euclidian_distances(euclid_dists,
                                                                                                nodes_data)

        self.graph = graph
        self.adjacency_matrix = adjacency_matrix
        self.matrix_connect = matrix_connect
        self.number_of_edges = number_of_edges

    def _load_graph(self, path):
        with open(path) as file:
            data = file.read()
            data = np.array(ast.literal_eval(data))
        self.graph = data

    def save_end_graph(self, name):
        res = []
        for i in range(self.number_of_nodes):
            try:
                val = self.graph[i]
            except:
                val = []
            res.append(val)
        with open(f"{self.cash_folder}/graph_{name}.txt", "w") as fl:
            fl.write(str(res))

    def add_edge(self, from_node: int, to_node: int):
        """
        Method for adding new edges.
        :param from_node:  start node of the edge
        :param to_node: end node of the edge
        """
        self.graph[from_node].append(to_node)
        self.number_of_edges += 1

    def remove_edge(self, from_node: int, to_node: int):
        """
        Method for removing edges.
        :param from_node:  start node of the edge
        :param to_node: end node of the edge
        """
        try:
            self.graph[from_node].remove(to_node)
        except:
            self.graph[to_node].remove(from_node)
        self.number_of_edges -= 1

    def twist_node(self, current_node: int):
        """
        Method for changing current node to another one from whole search space (source_data)
        :param current_node:  index of node to replace
        """
        all_nodes_indeces = np.arange(self.source_data.shape[0])
        available_nodes = np.delete(all_nodes_indeces, self.basis)
        new_node = available_nodes[np.random.choice(np.arange(available_nodes.shape[0]), size=1)[0]]
        self.basis[current_node] = new_node
        # recalculate distance between nodes
        euclid_dists = euclidean_distances(self.source_data[self.basis], self.source_data[self.basis])
        matrix_connect = euclid_dists / np.max(euclid_dists)
        self.matrix_connect = matrix_connect

    def get_start_node(self):
        """
        Method for searching node with maximum number of neighbours.
        The found node will be used as the starting point when filtering neighbors.
        :return  choose_index: int - index of the found node
        """
        choose_index = None
        for i, node in self.graph.items():
            try:
                if len(self.graph[choose_index]) < len(node):
                    choose_index = i
            except Exception:
                choose_index = i
        return choose_index

    def remove_edges(self, edges_list: list):
        """
        Method for removing multiple edges from the list.
        :param edges_list : list - tuples with start and end nodes of edges.
        """
        for edge in edges_list:
            if edge[0] not in self.graph[edge[1]] and edge[1] not in self.graph[edge[0]]:
                continue
            try:
                self.graph[edge[0]].remove(edge[1])
            except Exception as e:
                self.graph[edge[1]].remove(edge[0])
            self.number_of_edges -= 1

            if edge[0] in self.graph[edge[1]] or edge[1] in self.graph[edge[0]]:
                print("ux")

            if edge[0] in self.graph[edge[1]] or edge[1] in self.graph[edge[0]]:
                print("ux")

    def replace_subgraph(self, node: int, new_edges: list):
        """
        Method for replace some part of graph.
        :param node: int - index of the node whose connections with neighbours will be changed
        :param new_edges : list -  index of new neighbours for the node
        """
        self.number_of_edges -= len(self.graph[node])
        self.graph[node] = []
        for elem in new_edges:
            self.add_edge(node, elem)

    def check_vn_part(self, source_data: np.ndarray, node1: int, node2:int):
        """
        Method for check visible neighbours in new edge in graph added using crossover/evolution
        :param source_data : matrix n * m, where n - number of nodes, m - number of features. Keeping values of nodes by fields.
        :param node1 : one of nodes in the new edge
        :param node2 : one of nodes in the new edge
        """
        general_neighbours = []
        del_list = []
        gr1 = self.graph[node1]
        gr2 = self.graph[node2]
        for neigh in gr1:
            if neigh in gr2:
                general_neighbours.append(neigh)

        data_neigh = source_data[general_neighbours]
        dif_n1 = source_data[node1] - data_neigh
        dif_n2 = source_data[node1] - source_data[node2]

        result = np.diag(np.dot(dif_n1, dif_n2.T))
        for i, res in enumerate(result[result < 0]):
            del_list.append((node1, general_neighbours[i]))

        dif_n1 = source_data[node2] - data_neigh
        dif_n2 = source_data[node2] - source_data[node1]

        result = np.diag(np.dot(dif_n1, dif_n2.T))
        for i, res in enumerate(result[result < 0]):
            del_list.append((node2, general_neighbours[i]))

        self.remove_edges(del_list)

    @property
    def laplassian(self):
        laplassian = np.zeros_like(self.matrix_connect)
        temp = 1 - self.matrix_connect
        for key in self.graph:
            laplassian[[key], [self.graph[key]]] = temp[[key], [self.graph[key]]]
        return laplassian

    def calc_fullness(self):
        """
        Method for calculation the percentage of completion of the graph.
        """
        self.fullness = (len(list(filter(lambda elem: elem == 0, self.laplassian.reshape(-1)))) / 2 * 100) // len(
            self.laplassian.reshape(-1))

    def filter_graph(self, data: np.ndarray):
        """
        Method for filter the graph from unvisible neighbours.
        :param data: matrix n * m, where n - number of nodes, m - number of features. Keeping values of nodes by fields.
        """
        start_node_index = self.get_start_node()
        delete_edges = get_indices_to_del(data, self.adjacency_matrix, self.matrix_connect, start_node_index)
        self.remove_edges(delete_edges)


@njit
def get_indices_to_del(source_data, adjacency_matrix, eds, start_node):
    """
    Function for parallel calculation of unnecessary indices based on adjacency_matrix
    and euclidian distances between nodes

    :param source_data: list of nodes
    :param adjacency_matrix: adjacency matrix of graph
    :param eds: list with euclidian distances between nodes
    :param start_node: index of node to start filtering
    :return:
    """
    selects = np.zeros((len(source_data)))
    rem_edges = []
    start_nodes = [start_node]
    while len(start_nodes) > 0:
        current_node = start_nodes.pop(0)
        selects[current_node] = 1
        if sum(adjacency_matrix[current_node]) == 0:
            continue
        neigh_indexs = np.where(adjacency_matrix[current_node] == 1)[0]
        args = np.argsort(eds[current_node, neigh_indexs])
        neigh_indexs = neigh_indexs[args[::-1]]

        add_params = source_data[neigh_indexs]
        neighbours = source_data[current_node] - add_params

        for i, elem in enumerate(neigh_indexs):
            if selects[elem] == 1:
                continue
            check_this = source_data[elem]
            neigh_2 = check_this - add_params
            result = np.diag(np.dot(neighbours, neigh_2.T))
            if len(result[result < 0]) > 0:
                adjacency_matrix[current_node][elem] = 0
                adjacency_matrix[elem][current_node] = 0
                rem_edges.append((current_node, elem))
            else:
                start_nodes.append(elem)

    return rem_edges
