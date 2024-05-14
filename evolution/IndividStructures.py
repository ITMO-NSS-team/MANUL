import os
import pickle
from copy import deepcopy
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from datetime import datetime

import topo as tp
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go
import plotly


class DataStructureGraph:
    def __init__(self, data: np.ndarray = None,
                 n_neighbors: int = None,
                 epsilon_neighborhood: float = None,
                 graph_file: str = None,
                 cash_folder: str = None,
                 filter_obtuse_angles: bool = True):
        """
        Class for initialization individ  for evolution as complex graph structure with graph properties
        :param data: features table for graph structure creation
        :param n_neighbors:  number of neighbors for kernel fit (filtered laplacian)
        :param epsilon_neighborhood: epsilon distance between neighbors to decrease the closest
        :param graph_file: str - path to file .pkl with DataStructureGraph object
        :param cash_folder: str - path to save cash
        :param filter_obtuse_angles:  filter edges by geodesic distance comparison with euclidean when create the graph
        """
        self.elitism = False
        self.selected = False
        self.fitness = None
        if cash_folder is None:
            self.cash_folder = f"info_log/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"
        else:
            self.cash_folder = cash_folder

        if not os.path.exists(self.cash_folder):
            os.makedirs(self.cash_folder)
        print(f'Log folder set as {self.cash_folder}')

        if graph_file is not None:
            self.load_cash_object(graph_file)

        if graph_file is None and data is not None:
            self.source_data = data.astype(float)

            if n_neighbors is None:
                if data.shape[0] <= 500:
                    n_neighbors = 1
                if 500 < data.shape[0] <= 2000:
                    n_neighbors = 2
                if 2000 < data.shape[0]:
                    n_neighbors = 10
            else:
                n_neighbors = n_neighbors

            self.create_graph(data, n_neighbors, epsilon_neighborhood, filter_obtuse_angles)
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
        laplacian = self.laplacian
        if indices is not None:
            laplacian = laplacian[indices][:, indices]
        part_1 = np.dot(f_x.T, laplacian)
        loss = np.dot(part_1, f_x)
        return loss.reshape(-1)[0]

    def create_graph(self, nodes_data: np.ndarray,
                     n_neighbors: int,
                     epsilon_neighborhood: float,
                     filter_obtuse_angles: bool = True):
        """
        Method to create graph from table data
        :param filter_obtuse_angles: filter edges by geodesic distance comparison with euclidean
        :param epsilon_neighborhood: normalized to max distance threshold for long edges filtering
        :param n_neighbors: number of neighbors for kernel fit (filtered laplacian)
        :param nodes_data: matrix with features table data
        """
        euclid_dists = euclidean_distances(nodes_data, nodes_data)
        matrix_connect = euclid_dists / np.max(euclid_dists)

        kernel = tp.tpgraph.Kernel(n_neighbors=n_neighbors, n_jobs=1, metric='cosine', fuzzy=True,
                                   verbose=True)
        print(f'Fit kernel, n_neighbors={n_neighbors} ')
        kernel.fit(nodes_data)
        print(f'Laplacian calculation')
        filtered_lapl = kernel.L.todense()

        # filtering edges by filtered laplacian
        adjacency_matrix = np.zeros(euclid_dists.shape)
        adjacency_matrix[filtered_lapl != 0] = 1
        np.fill_diagonal(adjacency_matrix, 0)

        # get nodes pairs for edges
        edges = np.array(np.where(adjacency_matrix != 0))
        if filter_obtuse_angles:
            print(f'Geodesic distances calculation')
            geodesic_distances = kernel.SP
            valid_edges_inds = np.where(geodesic_distances[edges[0], edges[1]] >= euclid_dists[edges[0], edges[1]])[0]
            if valid_edges_inds.size != 0:
                edges = edges[:, valid_edges_inds]
        filtered_edges = edges[:, (edges[0] - edges[1]) != 0]

        # base is nodes which have at least one edge
        self.basis = np.unique(filtered_edges)
        adjacency_matrix = adjacency_matrix[self.basis][:, self.basis]

        if epsilon_neighborhood is None:
            # TODO  вынести параметр квартиля в гиперпараметры
            if nodes_data.shape[0] > 10000:
                quantile = 0.005
            elif 1000 <= nodes_data.shape[0] <= 10000:
                quantile = 0.1
            elif nodes_data.shape[0] < 1000:
                quantile = 0.2
            # filtering edges by saving quantile of all edged based on edge distance
            epsilon_neighborhood = np.round(np.quantile(matrix_connect, quantile), 2)
            print(f'epsilon_neighborhood = {epsilon_neighborhood}')

        # filtering too long edges (more than epsilon_neighborhood)
        euclid_dists = euclidean_distances(nodes_data[self.basis], nodes_data[self.basis])
        matrix_connect = euclid_dists / np.max(euclid_dists)
        euclid_adjacency_matrix = np.zeros(adjacency_matrix.shape)
        euclid_adjacency_matrix[matrix_connect <= epsilon_neighborhood] = 1
        np.fill_diagonal(euclid_adjacency_matrix, 0)

        self.adjacency_matrix = euclid_adjacency_matrix
        self.matrix_connect = matrix_connect

    def add_edges(self, edges_list: np.ndarray):
        """
        Method for adding new edges into adjacency_matrix of individ
        :param edges_list: pairwise indices of nodes to connect
        """
        self.adjacency_matrix[edges_list[0, :], edges_list[1, :]] = 1
        self.adjacency_matrix[edges_list[1, :], edges_list[0, :]] = 1

    def remove_edges(self, edges_list: np.ndarray):
        """
        Method for removing multiple edges from the list.
        :param edges_list : list - tuples with start and end nodes of edges.
        """
        self.adjacency_matrix[edges_list[0, :], edges_list[1, :]] = 0
        self.adjacency_matrix[edges_list[1, :], edges_list[0, :]] = 0

    def twist_nodes(self, current_nodes_indices: np.ndarray):
        """
        Method for changing list of selected nodes to others from whole search space (source_data)
        :param current_nodes_indices: indices from individ base to change
        """
        current_euclid_dists_max = np.max(
            euclidean_distances(self.source_data[self.basis], self.source_data[self.basis]))
        all_nodes_source_indeces = np.arange(self.source_data.shape[0])
        available_nodes = np.delete(all_nodes_source_indeces, self.basis)
        new_nodes_source_indices = available_nodes[np.random.choice(np.arange(available_nodes.shape[0]),
                                                                    size=current_nodes_indices.shape[0])]
        self.basis = np.array(self.basis)
        self.basis[current_nodes_indices] = new_nodes_source_indices
        new_nodes_source_matrix = self.source_data[self.basis[current_nodes_indices]]
        new_nodes_distances = euclidean_distances(new_nodes_source_matrix, new_nodes_source_matrix)
        new_euclid_dist_max = np.max(new_nodes_distances)
        new_nodes_distances = new_nodes_distances / np.max([new_euclid_dist_max, current_euclid_dists_max])
        self.matrix_connect[np.ix_(current_nodes_indices, current_nodes_indices)] = new_nodes_distances

    def change_edges_length(self, edges_inds: np.ndarray, mutate_intensity: float):
        """
        :param edges_inds: np array with integer indices of chosen edges
        :param mutate_intensity: float that shows the range of distance values changing
        """
        lengths_add = np.random.uniform(-mutate_intensity, mutate_intensity, edges_inds.shape[0])
        new_lengths = self.matrix_connect[edges_inds[:, 0], edges_inds[:, 1]] + lengths_add
        self.matrix_connect[edges_inds[:, 0], edges_inds[:, 1]] = new_lengths
        self.matrix_connect[edges_inds[:, 1], edges_inds[:, 0]] = new_lengths
        np.fill_diagonal(self.matrix_connect, 0)
        self.matrix_connect[self.matrix_connect > 1] = 1 - (self.matrix_connect[self.matrix_connect > 1] - 1)
        self.matrix_connect[self.matrix_connect < 0] = -self.matrix_connect[self.matrix_connect < 0]

    def replace_subgraph(self, node: int, new_edges: np.ndarray):
        """
        Method for replace some part of graph.
        :param node: int - index of the node whose connections with neighbours will be changed
        :param new_edges : indices of new neighbours for the node
        """
        self.adjacency_matrix[node] = 0
        self.adjacency_matrix[:, node] = 0
        self.adjacency_matrix[node][new_edges] = 1
        self.adjacency_matrix[new_edges, node] = 1


    @property
    def laplacian(self):
        """
        L=D-A, где D - степени вершин, а A - матрица весов
        Лапласиан как разница матрицы степеней вершин и матрицы весов,
        матрица степеней считается как сумма весов исходящих из вершины
        """
        weights_matrix = deepcopy(self.matrix_connect)
        weights_matrix[self.adjacency_matrix == 0] = 0
        nodes_weights = np.diag(np.sum(weights_matrix, axis=0))
        lap = nodes_weights - weights_matrix
        return lap

    @property
    def number_of_edges(self):
        """
        Property return number of edges in individ graph
        """
        return int(np.sum(self.adjacency_matrix) // 2)

    @property
    def number_of_nodes(self):
        """
        Property return number of nodes in individ graph
        """
        return self.adjacency_matrix.shape[0]


    def show_2d(self, labels: Optional[np.ndarray] = None,
                title: str = '',
                cmap_name: str = 'brg',
                save_path: str = None, euclidean=True):
        """
        Function to visualize individ graph structure in 2D projection
        :param save_path: string with path to save plot
        :param labels: array with target values of samples (nodes)
        :param title: string with name of plot
        :param cmap_name: string with Matplotlib colormap name
        :param euclidean: use adjacency_matrix for graph generation or euq
        """
        nodes_coordinates = self.source_data[self.basis]
        if nodes_coordinates.shape[1] > 2:
            pca = PCA(n_components=2)
            pca.fit(nodes_coordinates)
            nodes_coordinates = pca.transform(nodes_coordinates)

        fig, ax = plt.subplots()
        if euclidean:
            g = nx.Graph(self.adjacency_matrix)
        if not euclidean:
            g = nx.Graph(self.matrix_connect)
        if labels is not None:
            nodes_labels = labels[self.basis]
            colors = nodes_labels
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
            c = plt.colorbar(sm, ax=ax)
            c.set_label('Target values')
        else:
            colors = 'black'

        if euclidean:
            nx.draw_networkx_edges(g, pos=nodes_coordinates)
            n = nx.draw_networkx_nodes(g, pos=nodes_coordinates, linewidths=0.5, node_size=15, node_color=colors,
                                       cmap=cmap_name)
            n.set_edgecolor('black')

        if not euclidean:
            # drawing without fixed nodes position, but based on edges lengths
            nx.draw(g, node_color=colors, cmap=cmap_name, node_size=15, linewidths=0.1)

        fig.suptitle(title)
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        plt.show()

    def show_3d(self, labels: Optional[np.ndarray] = None,
                markers: Optional[np.ndarray] = None,
                title: str = None,
                save_path: str = None):

        nodes_coordinates = self.source_data[self.basis]
        initial_dims = nodes_coordinates.shape[1]
        if nodes_coordinates.shape[1] > 3:
            print(f'Computing PCA from {initial_dims} to 3')
            pca = PCA(n_components=3)
            pca.fit(nodes_coordinates)
            nodes_coordinates = pca.transform(nodes_coordinates)

        mask_matrix = np.full(self.adjacency_matrix.shape, 1)
        mask_matrix = np.tril(mask_matrix, -1)
        one_way_adj_matrix = mask_matrix * self.adjacency_matrix
        edges = np.where(one_way_adj_matrix == 1)

        start_nodes_positions = nodes_coordinates[edges[0], :]
        end_nodes_positions = nodes_coordinates[edges[1], :]

        Xe = []
        Ye = []
        Ze = []
        for n in range(start_nodes_positions.shape[0]):
            Xe += [start_nodes_positions[n][0], end_nodes_positions[n][0], None]  # x-coordinates of edge ends
            Ye += [start_nodes_positions[n][1], end_nodes_positions[n][1], None]
            Ze += [start_nodes_positions[n][2], end_nodes_positions[n][2], None]

        uniq_start_nodes_positions = nodes_coordinates[np.unique(edges[0]), :]

        if labels is not None:
            nodes_labels = labels[self.basis]
        else:
            nodes_labels = [1] * len(self.basis)

        if markers is not None:
            nodes_markers = markers[self.basis]
        else:
            nodes_markers = ['circle'] * len(self.basis)

        colors = nodes_labels
        trace1 = go.Scatter3d(x=Xe,
                              y=Ye,
                              z=Ze,
                              mode='lines',
                              line=dict(color='rgb(125,125,125)', width=1),
                              hoverinfo='none'
                              )

        trace2 = go.Scatter3d(x=uniq_start_nodes_positions[:, 0],
                              y=uniq_start_nodes_positions[:, 1],
                              z=uniq_start_nodes_positions[:, 2],
                              mode='markers',
                              name='actors',
                              marker=dict(symbol=(nodes_markers),
                                          size=6,
                                          color=colors,
                                          colorscale='Viridis',
                                          line=dict(color='rgb(50,50,50)', width=0.5)
                                          ),
                              text=nodes_labels,
                              hoverinfo='text'
                              )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=True,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
            title=f'{title}\nProjection of {initial_dims} features to 3d',
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest',
        )
        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        if not save_path:
            if title is None:
                title = '3d_graph'
            save_path = f'{self.cash_folder}/{title}.html'
        plotly.offline.plot(fig, filename=save_path)

