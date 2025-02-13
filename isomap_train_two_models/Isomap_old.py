import torch
from torch import nn

device = 'cuda'

class FloydWarshall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph):
        """
        Forward pass for Floyd-Warshall algorithm.
        """
        n_samples = graph.size(0)
        dist = graph.clone()

        # Run Floyd-Warshall algorithm
        for k in range(n_samples):
            dist = torch.min(dist, dist[:, k:k+1] + dist[k:k+1, :])
        #dist = torch.min(dist, dist[:, :, None] + dist[None, :, :])
        # Save necessary tensors for the backward pass
        ctx.save_for_backward(graph, dist)
        return dist

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradients efficiently.
        """
        graph, dist = ctx.saved_tensors
        n_samples = graph.size(0)

        # Initialize gradient w.r.t. the input graph
        grad_input = torch.zeros_like(graph)

        # Compute gradients by propagating through the relaxation steps
        for k in reversed(range(n_samples)):
            grad_input += (grad_output < (dist[:, k:k+1] + dist[k:k+1, :])).float() * grad_output

        return grad_input



class TestPointShortestPaths(torch.autograd.Function):
    @staticmethod
    def forward(ctx, test_dist, dist_matrix, n_neighbors):
        """
        Forward pass to compute shortest paths for test points to training points.
        """
        n_test, n_train = test_dist.size()

        # Identify the indices of the n_neighbors closest training points for each test point
        neighbors = torch.topk(test_dist, n_neighbors, largest=False).indices  # Shape: (n_test, n_neighbors)

        # Gather the distances for the selected neighbors
        neighbor_distances = test_dist.gather(1, neighbors)  # Shape: (n_test, n_neighbors)
        dist_matrix_neighbors = dist_matrix[neighbors]  # Shape: (n_test, n_neighbors, n_train)

        # Compute the combined distances from test points through neighbors to training points
        combined_distances = dist_matrix_neighbors + neighbor_distances.unsqueeze(-1)  # Shape: (n_test, n_neighbors, n_train)

        # Compute the minimum distances for each test point to training points
        G_X, min_indices = combined_distances.min(dim=1)  # G_X: (n_test, n_train), min_indices: (n_test, n_train)

        # Store necessary variables for backward pass
        ctx.save_for_backward(test_dist, dist_matrix, neighbors, min_indices)
        ctx.n_neighbors = n_neighbors

        return G_X

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradients for test_dist and dist_matrix.
        """
        test_dist, dist_matrix, neighbors, min_indices = ctx.saved_tensors
        n_neighbors = ctx.n_neighbors
        n_test, n_train = test_dist.size()

        # Initialize gradients
        grad_test_dist = torch.zeros_like(test_dist)
        grad_dist_matrix = torch.zeros_like(dist_matrix)

        # Backpropagate gradients
        # Iterate over each test point
        for i in range(n_test):
            # Get the neighbors and corresponding indices
            neighbor_indices = neighbors[i]  # Shape: (n_neighbors,)
            min_idx = min_indices[i]  # Shape: (n_train,)

            # Scatter gradients for the neighbors
            for j in range(n_train):
                neighbor = neighbor_indices[min_idx[j]]
                grad_test_dist[i, neighbor] += grad_output[i, j]
                grad_dist_matrix[neighbor, j] += grad_output[i, j]

        return grad_test_dist, grad_dist_matrix, None



class IsomapNN(nn.Module):
    def __init__(self, weights_initial_assumption: torch.tensor,
                 n_components: int = 2,
                 n_neighbors: int = 25):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.distances_matrix = weights_initial_assumption
        self._init_weights(weights_initial_assumption)
        #self.dist_matrix_ = None

    def update_distance_matrix(self):
        matrix = torch.zeros(self.distances_matrix.size(0), self.distances_matrix.size(1)).to(device)
        matrix[self.params_inds_in_matrix[:, 0], self.params_inds_in_matrix[:, 1]] = abs(self.layer)
        matrix[self.params_inds_in_matrix[:, 1], self.params_inds_in_matrix[:, 0]] = abs(self.layer)
        #matrix[self.params_inds_in_matrix[:, 0], self.params_inds_in_matrix[:, 1]] = self.layer
        #matrix[self.params_inds_in_matrix[:, 1], self.params_inds_in_matrix[:, 0]] = self.layer
        self.distances_matrix = matrix

    def _init_weights(self, distances_matrix):
        upper_diag = torch.triu(distances_matrix, diagonal=1)
        params_inds = torch.nonzero(upper_diag)
        self.params_inds_in_matrix = params_inds
        values = upper_diag[params_inds[:, 0], params_inds[:, 1]]
        self.layer = torch.nn.Parameter(values)

    def _compute_shortest_paths_simple(self, graph):
        """Compute shortest paths using Floyd-Warshall algorithm."""
        n_samples = graph.size(0)
        dist = graph.clone()
        for k in range(n_samples):
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        return dist



    def _compute_shortest_paths_optimized(self, graph):
        """
        Wrapper for memory-efficient shortest path computation.
        """
        return FloydWarshall.apply(graph)

    def _compute_shortest_paths(self, graph, optimized=True):
        """Compute shortest paths using Floyd-Warshall algorithm."""
        if optimized:
            dist = self._compute_shortest_paths_optimized(graph)
        else:
            dist = self._compute_shortest_paths_simple(graph)
        return dist

    def _compute_test_shortest_paths_simple(self, test_distances):
        n_test = test_distances.size(0)
        n_train = self.dist_matrix_.size(0)
        G_X = torch.full((n_test, n_train), float('inf'), dtype=test_distances.dtype).to(device)
        for i in range(n_test):
            distances = test_distances[i]
            neighbors = torch.topk(distances, self.n_neighbors, largest=False).indices
            for neighbor in neighbors:
                G_X[i] = torch.minimum(G_X[i], self.dist_matrix_[neighbor] + distances[neighbor])
        return G_X

    def _compute_test_shortest_paths_optimized(self, test_distances):
        """
        Wrapper for memory-efficient test shortest path computation.
        """
        return TestPointShortestPaths.apply(test_distances, self.dist_matrix_, self.n_neighbors)

    def _compute_test_shortest_paths(self,test_distances,optimized=True):
        '''Compute shortest paths for test points to training points'''
        if optimized:
            dist = self._compute_test_shortest_paths_optimized(test_distances)
        else:
            dist = self._compute_test_shortest_paths_simple(test_distances)
        return dist

    def _construct_graph(self, distance_matrix):
        """Construct a graph where only the distances to neighbors are preserved."""
        n_samples = distance_matrix.size(0)
        graph = torch.full_like(distance_matrix, float('inf'))
        for i in range(n_samples):
            distances = distance_matrix[i]
            neighbors = torch.topk(distances, self.n_neighbors, largest=False).indices
            graph[i, neighbors] = distances[neighbors]
        graph = torch.min(graph, graph.T)  # Ensure symmetry
        return graph

    def _double_centering(self, distances):
        """Perform double centering on the distance matrix."""
        n_samples = distances.size(0)
        H = torch.eye(n_samples, device=distances.device) - (1 / n_samples) * torch.ones((n_samples, n_samples), device=distances.device)
        K = -0.5 * H @ (distances ** 2) @ H
        return K

    def fit_transform(self, distance_matrix):
        """Fit the model and return the embedding for the training set."""
        graph = self._construct_graph(distance_matrix)
        geodesic_distances = self._compute_shortest_paths(graph)
        self.dist_matrix_ = geodesic_distances.clone()

        # Create kernel and perform eigen decomposition
        K = self._double_centering(geodesic_distances)
        eigenvalues, eigenvectors = torch.linalg.eigh(K)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(torch.abs(eigenvalues), descending=True)
        eigenvalues = eigenvalues[sorted_indices][:self.n_components]
        eigenvectors = eigenvectors[:, sorted_indices][:, :self.n_components]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.embedding_ = self.eigenvectors_*torch.sign(self.eigenvalues_) * torch.sqrt(torch.abs(self.eigenvalues_))

        return self.embedding_

    def transform(self, test_distances):
        """Transform new points based on the fitted Isomap model."""
        if self.embedding_ is None or self.eigenvalues_ is None or self.eigenvectors_ is None:
            raise ValueError("The model must be fitted before calling transform.")

        # n_test = test_distances.size(0)
        # n_train = self.dist_matrix_.size(0)

        # # Compute shortest paths for test points to training points
        # G_X = torch.full((n_test, n_train), float('inf'), dtype=test_distances.dtype).to(device)
        # for i in range(n_test):
        #     distances = test_distances[i]
        #     neighbors = torch.topk(distances, self.n_neighbors, largest=False).indices
        #     for neighbor in neighbors:
        #         G_X[i] = torch.min(G_X[i], self.dist_matrix_[neighbor] + distances[neighbor])

        G_X = self._compute_test_shortest_paths(test_distances,optimized=True)

        # Center the distances for test points
        train_mean = torch.mean(self.dist_matrix_ ** 2, dim=1, keepdim=True)
        overall_mean = torch.mean(self.dist_matrix_ ** 2)
        G_X **= 2
        K_test = -0.5 * (G_X - train_mean.T - torch.mean(G_X, dim=1, keepdim=True) + overall_mean)

        # Project test points into the embedding space
        test_embedding = K_test @ self.eigenvectors_*torch.sign(self.eigenvalues_) / torch.sqrt(torch.abs(self.eigenvalues_)+ 1e-8)
        return test_embedding

    def forward(self):
        self.update_distance_matrix()
        transformed_points = self.fit_transform(self.distances_matrix)
        return transformed_points
