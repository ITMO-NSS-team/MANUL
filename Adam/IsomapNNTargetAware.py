import torch
from torch import nn
from Adam.KernelPCA import KernelPCA


device = 'cuda'


def smacof_single_pytorch(
    dissimilarities,
    n_components=2,
    init=None,
    max_iter=300,
    eps=1e-3,
    random_state=None,
    device='cpu',
    verbose=0
):
    D = dissimilarities.to(device).double()  # Force double precision
    n_samples = D.shape[0]

    if random_state is not None:
        torch.manual_seed(random_state)

    # Initialize and scale X to match D's magnitude
    if init is None:
        X = torch.rand(n_samples, n_components, device=device, dtype=torch.double)
        with torch.no_grad():
            S_init = torch.cdist(X, X, p=2)
            scale = (D.mean() / S_init.mean()).item()
            X *= scale
    else:
        X = init.to(device).double()

    old_stress = None
    for it in range(max_iter):
        S = torch.cdist(X, X, p=2) + 1e-6  # Add epsilon to avoid division by zero
        stress = 0.5 * torch.sum((S - D)**2)

        # Guttman transform
        ratio = D / S
        B = -ratio
        diag_updates = ratio.sum(dim=1)
        B.diagonal().add_(diag_updates)

        X_new = (1.0 / n_samples) * (B @ X)

        # Update X and compute normalization
        X = X_new
        dis = torch.norm(X, dim=1).sum()
        if dis < 1e-6:
            dis = torch.tensor(1e-6, device=device)

        normalized_stress = stress / dis
        if verbose >= 2:
            print(f"Iter {it}: Stress = {stress.item()}")

        # Check convergence
        if old_stress is not None and (old_stress - normalized_stress) < eps:
            if verbose:
                print(f"Converged at iteration {it}")
            break
        old_stress = normalized_stress

    return X


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

        grad_input = torch.zeros_like(graph)

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

        neighbors = torch.topk(test_dist, n_neighbors, largest=False).indices  # Shape: (n_test, n_neighbors)

        neighbor_distances = test_dist.gather(1, neighbors)  # Shape: (n_test, n_neighbors)
        dist_matrix_neighbors = dist_matrix[neighbors]  # Shape: (n_test, n_neighbors, n_train)

        combined_distances = dist_matrix_neighbors + neighbor_distances.unsqueeze(-1)  # Shape: (n_test, n_neighbors, n_train)

        G_X, min_indices = combined_distances.min(dim=1)  # G_X: (n_test, n_train), min_indices: (n_test, n_train)

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

        grad_test_dist = torch.zeros_like(test_dist)
        grad_dist_matrix = torch.zeros_like(dist_matrix)

        for i in range(n_test):
            neighbor_indices = neighbors[i]  # Shape: (n_neighbors,)
            min_idx = min_indices[i]  # Shape: (n_train,)

            for j in range(n_train):
                neighbor = neighbor_indices[min_idx[j]]
                grad_test_dist[i, neighbor] += grad_output[i, j]
                grad_dist_matrix[neighbor, j] += grad_output[i, j]

        return grad_test_dist, grad_dist_matrix, None


class IsomapNNTargetAware(nn.Module):
    def __init__(self, weights_initial_assumption: torch.tensor,
                 n_components: int = 2,
                 n_neighbors: int = 25,
                 eigval_choice: str = 'MDS',
                 # Target-aware parameters
                 targets: torch.Tensor = None,
                 use_target_modification: bool = False,
                 target_weight: float = 1.0,
                 epsilon: float = 0.01,
                 modification_mode: str = 'l2'):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.distances_matrix = weights_initial_assumption

        # Target-aware attributes
        self.use_target_modification = use_target_modification
        self.target_weight = target_weight
        self.epsilon = epsilon
        self.modification_mode = modification_mode

        # Compute target_diff once and register as buffer
        if use_target_modification and targets is not None:
            if targets.dim() == 2 and targets.shape[1] > 1:
                # Multi-dim targets (e.g., one-hot): pairwise L2 distance
                target_diff = torch.cdist(targets.float(), targets.float())
            else:
                # Scalar targets: absolute difference
                targets = targets.flatten().float()
                target_diff = torch.abs(targets.unsqueeze(0) - targets.unsqueeze(1))
            max_diff = target_diff.max()
            if max_diff > 0:
                target_diff = target_diff / max_diff
            self.register_buffer('target_diff', target_diff)
        else:
            self.register_buffer('target_diff', None)

        self._init_weights(weights_initial_assumption)
        self.kernel_pca_ = KernelPCA(n_components=self.n_components,eigval_choice=eigval_choice)

    def update_distance_matrix(self):
        matrix = torch.zeros(self.distances_matrix.size(0), self.distances_matrix.size(1)).to(device)
        matrix[self.params_inds_in_matrix[:, 0], self.params_inds_in_matrix[:, 1]] = abs(self.layer)
        matrix[self.params_inds_in_matrix[:, 1], self.params_inds_in_matrix[:, 0]] = abs(self.layer)

        # Target-aware distance modification
        if self.use_target_modification and self.target_diff is not None:
            if self.modification_mode == 'multiply':
                # d_new = d * |Δy| * λ + ε
                matrix = matrix * self.target_diff * self.target_weight + self.epsilon
            elif self.modification_mode == 'add':
                # d_new = d + λ * |Δy|
                matrix = matrix + self.target_weight * self.target_diff
            elif self.modification_mode == 'l2':
                # d_new = sqrt(d² + λ * |Δy|²)
                matrix = torch.sqrt(matrix ** 2 + self.target_weight * self.target_diff ** 2)
            else:
                raise ValueError(f"Unknown modification_mode: {self.modification_mode}")

            # Ensure zero diagonal
            matrix = matrix.clone()
            matrix.fill_diagonal_(0)

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

    def fit_transform(self, distance_matrix, SMACOF_refine = False):
        """Fit the model and return the embedding for the training set."""
        graph = self._construct_graph(distance_matrix)
        geodesic_distances = self._compute_shortest_paths(graph)

        geodesic_distances = torch.nan_to_num(geodesic_distances, nan=0.0, posinf=0.0, neginf=0.0)

        # Add small epsilon to avoid completely zero distances
        geodesic_distances = geodesic_distances + 1e-8

        self.dist_matrix_ = geodesic_distances.clone()

        G = self.dist_matrix_**2
        G *= -0.5

        G_reg = G + torch.eye(G.size(0), device=G.device) * 1e-6

        self.embedding_ = self.kernel_pca_.fit_transform(G_reg)

        if SMACOF_refine:
            self.refined_embedding_ = smacof_single_pytorch(self.distances_matrix, n_components=self.n_components,init=self.embedding_,device=G.device)
            return self.refined_embedding_
        else:
            return self.embedding_

    def transform(self, test_distances,SMACOF_refine = False):

        G_X = self._compute_test_shortest_paths(test_distances,optimized=True)

        if SMACOF_refine:
            diss = G_X.clone()

        # Center the distances for test points
        train_mean = torch.mean(self.dist_matrix_ ** 2, dim=1, keepdim=True)
        overall_mean = torch.mean(self.dist_matrix_ ** 2)
        G_X **= 2
        G_X *= -0.5


        if SMACOF_refine:
            self.refined_embedding_ = smacof_single_pytorch(diss, n_components=self.n_components,init=self.kernel_pca_.transform(G_X),device=G_X.device)
            return self.refined_embedding_
        else:
            return self.kernel_pca_.transform(G_X)

    def forward(self):
        self.update_distance_matrix()
        transformed_points = self.fit_transform(self.distances_matrix)
        return transformed_points