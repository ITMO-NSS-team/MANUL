import numpy as np
import torch
from matplotlib import pyplot as plt

from Adam.Isomap import IsomapNN


class DataStructureGraph:
    def __init__(self, dimensionality: int,
                 targets: np.ndarray,
                 distances_matrix: np.ndarray = None,
                 features: np.ndarray = None
                 ):

        self.features = None
        self.targets = targets
        self.dimensionality = dimensionality
        self._init_structure(distances_matrix, features)

        self.elitism = False
        self.selected = False
        self.fitness = None
        self.criteria = None
        self.level = None

    def _init_structure(self, distances_matrix: np.ndarray, features: np.ndarray):
        if distances_matrix is None:
            # determine matrix by projection (actual for base individ)
            print('Init distance matrix for initial assumption')
            #self.distances_matrix = pairwise_distances(features, features)
            self.distances_matrix = self.generate_random_matrix(features.shape[0], dist_type='normal')
        if distances_matrix is not None:
            self.distances_matrix = distances_matrix
        self.upd_features()

    @staticmethod
    def generate_random_matrix(n_samples, dist_type='normal'):
        if dist_type == 'uniform':
            matrix = np.random.rand(n_samples, n_samples)
        elif dist_type == 'normal':
            matrix = np.abs(np.random.randn(n_samples, n_samples))
        elif dist_type == 'exp':
            matrix = np.random.rand(n_samples, n_samples) ** 2

        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        return matrix / matrix.max()

    def upd_features(self):
        """Building features projection from distance matrix via Isomap"""
        isomap_model = IsomapNN(torch.Tensor(self.distances_matrix).to('cuda'), n_components=self.dimensionality, eigval_choice='MDS')
        features = isomap_model()
        self.features = features.cpu().detach().numpy()

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
        new_edges = new_edges.flatten()
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
    def eigenvalues(self):
        return np.linalg.eigh(self.distances_matrix)[0][:2]

    @property
    def valid_eigenvalues(self):
        return abs(self.eigenvalues[0] - self.eigenvalues[1]) >= 0.1

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

    def visualize(self, save_path: str = None):
        if self.features.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(self.features)
            explained_var = pca.explained_variance_ratio_.sum()
            projection_info = f"\n(PCA: {explained_var:.3f} variance explained)"
        else:
            features_2d = self.features
            projection_info = ''

        plt.figure(figsize=(6, 4))

        plt.scatter(features_2d[:, 0], features_2d[:, 1],
                    s=5, alpha=0.7, c=self.targets)

        rows, cols = np.where(np.triu(self.distances_matrix) > 0)

        if len(rows) > 0:
            from matplotlib.collections import LineCollection
            segments = np.array([[features_2d[i], features_2d[j]] for i, j in zip(rows, cols)])
            lc = LineCollection(segments, colors='gray', alpha=0.1, linewidths=0.5)
            plt.gca().add_collection(lc)

        plt.title(
            f'Graph: {self.number_of_nodes} nodes, {self.number_of_edges} edges{projection_info}, \neigh {self.eigenvalues}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f'{save_path}')
            plt.close()
        else:
            plt.show()
