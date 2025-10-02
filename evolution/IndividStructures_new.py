
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import Isomap


class DataStructureGraph:
    def __init__(self, dimensionality: int,
                 distances_matrix: np.ndarray = None,
                 features: np.ndarray = None
                 ):

        self.dimensionality = dimensionality
        self._init_structure(distances_matrix, features)

        self.elitism = False
        self.selected = False
        self.fitness = None
        self.criteria = None
        self.level = None

    def _init_structure(self, distances_matrix: np.ndarray, features: np.ndarray):
        if distances_matrix is None and features is None:
            raise Exception('No data for building graph! distances_matrix or features should be specified!')
        if distances_matrix is None:
            # determine matrix by projection (actual for base individ)
            print('Init graph as initial assumption on euclidean distances')
            self.distances_matrix = pairwise_distances(features, features)
        if distances_matrix is not None:
            self.distances_matrix = distances_matrix
            if features is not None:
                print('Distance matrix is prior for graph building, specified features are ignored')
            self.upd_features()

    def upd_features(self):
        """Building features projection from distance matrix via Isomap"""
        isomap = Isomap(n_components=self.dimensionality, n_neighbors=15, metric='precomputed')
        features = isomap.fit_transform(self.distances_matrix)
        self.features = features

    def add_edges(self, edges_list: np.ndarray):
        """
        Method for adding new edges into adjacency_matrix of individ
        :param edges_list: numpy array of shape (2, N) where:
        - edges_list[0, :] contains start node indices
        - edges_list[1, :] contains end node indices
        Each column represents an edge (start, end) to add
        """
        start_nodes = edges_list[0, :]
        end_nodes = edges_list[1, :]

        distances = np.linalg.norm(self.features[start_nodes] - self.features[end_nodes], axis=1)

        self.distances_matrix[start_nodes, end_nodes] = distances
        self.distances_matrix[end_nodes, start_nodes] = distances

    def remove_edges(self, edges_list: np.ndarray):
        """
        Method for removing multiple edges from the list.
        :param edges_list: numpy array of shape (2, N) where:
        - edges_list[0, :] contains start node indices
        - edges_list[1, :] contains end node indices
        Each column represents an edge (start, end) to remove
        """
        self.distances_matrix[edges_list[0, :], edges_list[1, :]] = 0
        self.distances_matrix[edges_list[1, :], edges_list[0, :]] = 0

    def change_edges_length(self, edges_inds: np.ndarray, mutate_intensity: float):
        """
        :param edges_inds: np array with integer indices of chosen edges
        :param mutate_intensity: float that shows the range of distance values changing
        """
        lengths_add = np.random.uniform(-mutate_intensity, mutate_intensity, edges_inds.shape[1])
        new_lengths = self.distances_matrix[edges_inds[0], edges_inds[1]] + lengths_add
        self.distances_matrix[edges_inds[0], edges_inds[1]] = new_lengths
        self.distances_matrix[edges_inds[1], edges_inds[0]] = new_lengths
        np.fill_diagonal(self.distances_matrix, 0)
        self.distances_matrix[self.distances_matrix > 1] = 1 - (self.distances_matrix[self.distances_matrix > 1] - 1)
        self.distances_matrix[self.distances_matrix < 0] = -self.distances_matrix[self.distances_matrix < 0]

    def replace_subgraph(self, node: int, new_edges: np.ndarray):
        """
        Method for replace some part of graph.
        :param node: int - index of the node whose connections with neighbours will be changed
        :param new_edges : indices of new neighbours for the node
        """
        # Remove all existing connections to this node
        self.distances_matrix[node, :] = 0
        self.distances_matrix[:, node] = 0

        # Calculate distances to new neighbors
        node_position = self.features[node]  # Position of the central node
        new_neighbors_positions = self.features[new_edges]  # Positions of new neighbors

        # Calculate Euclidean distances
        distances = np.linalg.norm(new_neighbors_positions - node_position, axis=1)

        # Add new edges with calculated distances
        self.distances_matrix[node, new_edges] = distances
        self.distances_matrix[new_edges, node] = distances

    @property
    def number_of_edges(self):
        """
        Property return number of edges in individ graph
        """
        return int(np.sum([self.distances_matrix != 0]) // 2)

    @property
    def number_of_nodes(self):
        """
        Property return number of nodes in individ graph
        """
        return self.distances_matrix.shape[0]

    def visualize(self, alpha=0.7, s=2, line_alpha=0.1):
        """
        Lightweight visualization for large graphs using pure matplotlib
        """
        if self.features.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(self.features)
            explained_var = pca.explained_variance_ratio_.sum()
            projection_info = f"\n(PCA: {explained_var:.3f} variance explained)"
        else:
            features_2d = self.features
            projection_info = ''

        plt.figure(figsize=(10, 5))
        plt.scatter(features_2d[:, 0], features_2d[:, 1],
                    s=s, alpha=alpha, c='red')

        rows, cols = np.where(np.triu(self.distances_matrix) > 0)
        for i, j in zip(rows, cols):
            plt.plot([features_2d[i, 0], features_2d[j, 0]],
                     [features_2d[i, 1], features_2d[j, 1]],
                     'gray', alpha=line_alpha, linewidth=0.5)

        plt.title(f'Graph Visualization: {self.number_of_nodes} nodes, {self.number_of_edges}{projection_info}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
