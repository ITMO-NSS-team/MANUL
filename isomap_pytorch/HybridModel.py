import torch
from torch import nn, float32

device = 'cuda'


class Hybrid(nn.Module):
    def __init__(self, weights_initial_assumption: torch.tensor, input_dim):
        super().__init__()
        self.n_components = 2  # isomap param
        self.n_neighbors = 25  # isomap param
        self.distances_matrix = weights_initial_assumption
        self._init_weights(weights_initial_assumption)

        self.task_model = nn.Sequential(*[nn.Linear(input_dim, 512, dtype=float32),
                                          nn.Linear(512, 256, dtype=float32),
                                          nn.Linear(256, 64, dtype=float32),
                                          nn.Linear(64, 1, dtype=float32)])
        self.default_task_weights = self.task_model.state_dict()

    def update_distance_matrix(self):
        matrix = torch.zeros(self.distances_matrix.size(0), self.distances_matrix.size(1)).to(device)
        matrix[self.params_inds_in_matrix[:, 0], self.params_inds_in_matrix[:, 1]] = abs(self.isomap_params)
        matrix[self.params_inds_in_matrix[:, 1], self.params_inds_in_matrix[:, 0]] = abs(self.isomap_params)
        self.distances_matrix = matrix

    def _init_weights(self, distances_matrix):
        upper_diag = torch.triu(distances_matrix, diagonal=1)
        params_inds = torch.nonzero(upper_diag)
        self.params_inds_in_matrix = params_inds
        values = upper_diag[params_inds[:, 0], params_inds[:, 1]]
        self.isomap_params = torch.nn.Parameter(values)

    def _floyd_warshall(self, graph: torch.tensor):
        """
        Perform the Floyd-Warshall algorithm to compute shortest paths in the graph.
        Args:
            graph (torch.Tensor): Adjacency matrix of the graph (n_samples, n_samples).
        Returns:
            torch.Tensor: Matrix of shortest path distances (n_samples, n_samples).
        """
        n_samples = graph.size(0)
        dist = graph.clone()
        for k in range(n_samples):
            dist = torch.min(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
        return dist

    def _randomized_eigen_decomposition(self, K, n_components, n_oversamples=10, n_iter=2):
        """
        Perform randomized eigen decomposition on the kernel matrix K.
        Args:
            K (torch.Tensor): Kernel matrix (n_samples, n_samples).
            n_components (int): Number of eigenvalues and eigenvectors to compute.
            n_oversamples (int): Additional dimensions for better approximation.
            n_iter (int): Number of power iterations for improved accuracy.

        Returns:
            (torch.Tensor, torch.Tensor): Eigenvalues and eigenvectors.
        """
        n_samples = K.size(0)
        Q = torch.randn(n_samples, n_components + n_oversamples, device=K.device)
        for _ in range(n_iter):
            Q = K @ Q
            Q = Q / torch.linalg.norm(Q, dim=0)
        B = Q.T @ K @ Q  # Compressed matrix
        # Ensure symmetry and add regularization
        B = (B + B.T) / 2
        regularization = 1e-6 * torch.eye(B.size(0), device=B.device)
        B += regularization
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(B)
        except torch._C._LinAlgError:
            print("Eigen decomposition failed, falling back to SVD.")
            U, S, V = torch.linalg.svd(B)
            eigenvalues = S[:n_components]
            eigenvectors = U[:, :n_components]
        # Sort and map back to original space
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices][:n_components]
        eigenvectors = eigenvectors[:, sorted_indices][:, :n_components]
        eigenvectors = Q @ eigenvectors

        return eigenvalues, eigenvectors

    def _fit_isomap(self, distance_matrix):
        n_samples = distance_matrix.size(0)
        # if (distance_matrix < 0).any():
        # raise ValueError("The distance matrix contains negative values, which are not allowed.")
        #self.training_distances_ = torch.clamp(distance_matrix, min=0)  # Ensure non-negativity
        self.training_distances_ = distance_matrix
        # Create a k-nearest neighbors graph
        graph = torch.full_like(self.training_distances_, float('inf'))
        for i in range(n_samples):
            distances = self.training_distances_[i]
            neighbors = torch.topk(distances, self.n_neighbors, largest=False).indices
            graph[i, neighbors] = distances[neighbors]
        graph = torch.min(graph, graph.T)  # Ensure symmetry
        # Compute geodesic distances
        geodesic_distances = self._floyd_warshall(graph)
        # Double centering to create kernel matrix
        H = torch.eye(n_samples, device=distance_matrix.device) - (1 / n_samples) * torch.ones((n_samples, n_samples),
                                                                                               device=distance_matrix.device)
        K = -0.5 * H @ (geodesic_distances ** 2) @ H
        # Randomized eigen decomposition
        eigenvalues, eigenvectors = self._randomized_eigen_decomposition(K, self.n_components)
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        # Compute the embedding
        self.embedding_ = self.eigenvectors_ * torch.sqrt(self.eigenvalues_)
        return self.embedding_.to(torch.float32)

    '''def predict_isomap(self, test_distances):
        if self.embedding_ is None:
            raise ValueError("The model must be fit before calling transform.")

        n_test = test_distances.size(0)
        n_train = self.training_distances_.size(0)

        # Initialize G_X with inf values
        G_X = torch.full((n_test, n_train), float('inf'), dtype=test_distances.dtype, device=test_distances.device)

        for i in range(n_test):
            # Find nearest neighbors of the test point among training points
            distances = test_distances[i]
            neighbors = torch.topk(distances, self.n_neighbors, largest=False).indices

            # Update geodesic distances via training graph
            for neighbor in neighbors:
                # Compute new distances without modifying G_X in place
                new_distances = self.training_distances_[neighbor] + distances[neighbor]
                G_X[i] = torch.minimum(G_X[i], new_distances)

        # Construct the test kernel
        train_mean = torch.mean(self.training_distances_ ** 2, dim=1, keepdim=True)
        overall_mean = torch.mean(self.training_distances_ ** 2)

        # Use G_X without in-place modification
        G_X_squared = G_X ** 2  # Create a new tensor for G_X squared
        K_test = -0.5 * (G_X_squared - train_mean.T - torch.mean(G_X_squared, dim=1, keepdim=True) + overall_mean)

        K_test = K_test.to(torch.float32)

        # Project test points onto the training eigenvectors
        test_embedding = K_test @ self.eigenvectors_ / torch.sqrt(self.eigenvalues_)

        return test_embedding'''

    def forward(self, points=None, isomap_step=False):
        if isomap_step:
            self.update_distance_matrix()
            transformed_points = self._fit_isomap(self.distances_matrix)
            self.task_model.load_state_dict(self.default_task_weights)
            self.task_model.train()
            return transformed_points
        else:
            if points is None:
                raise Exception('No points')
            out = self.task_model(points)
            return out
