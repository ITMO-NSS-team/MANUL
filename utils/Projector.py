import os
import json
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from Adam.Isomap import IsomapNN


def project_krr_optimized(X_sparse, Y_sparse, X_all, batch_size=1000):
    """Optimized Kernel Ridge Regression"""
    param_grid = {'alpha': [0.1, 1.0, 10.0], 'gamma': [0.01, 0.1, 1.0]}
    krr = KernelRidge(kernel='rbf')

    if len(X_sparse) > 1000:
        subset_idx = np.random.choice(len(X_sparse), 1000, replace=False)
        X_tune, Y_tune = X_sparse[subset_idx], Y_sparse[subset_idx]
    else:
        X_tune, Y_tune = X_sparse, Y_sparse

    grid_search = GridSearchCV(krr, param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_tune, Y_tune)
    best_krr = grid_search.best_estimator_
    best_krr.fit(X_sparse, Y_sparse)

    Y_projected = []
    for i in range(0, len(X_all), batch_size):
        batch = X_all[i:i + batch_size]
        Y_projected.append(best_krr.predict(batch))

    return np.concatenate(Y_projected, axis=0)


def project_ensemble_knn(X_sparse, Y_sparse, X_all):
    """Ensemble KNN regression"""
    estimators = [
        ('knn5', KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)),
        ('knn10', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)),
        ('knn15', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)),
    ]

    n_components = Y_sparse.shape[1]
    Y_projected = np.zeros((len(X_all), n_components))

    for dim in range(n_components):
        ensemble = VotingRegressor(estimators, n_jobs=-1)
        ensemble.fit(X_sparse, Y_sparse[:, dim])
        Y_projected[:, dim] = ensemble.predict(X_all)

    return Y_projected


def project_random_forest(X_sparse, Y_sparse, X_all, batch_size=1000):
    """Random Forest regression"""
    n_components = Y_sparse.shape[1]
    Y_projected = np.zeros((len(X_all), n_components))

    for dim in range(n_components):
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=None,
            min_samples_split=5, n_jobs=-1, random_state=42
        )
        rf.fit(X_sparse, Y_sparse[:, dim])

        dim_pred = []
        for i in range(0, len(X_all), batch_size):
            batch = X_all[i:i + batch_size]
            dim_pred.append(rf.predict(batch))

        Y_projected[:, dim] = np.concatenate(dim_pred)

    return Y_projected


class Projector:
    """
    Class for computing and managing projections from Euclidean space to hidden geometry space.

    This class handles:
    1. Computing base projections via Isomap (or using precomputed ones)
    2. Computing projections for all points via KNN interpolation
    3. Saving/loading projections to/from disk
    """

    @staticmethod
    def reconstruct_distance_matrix(upper_triangular_distances: np.ndarray, n_basis: int) -> np.ndarray:
        """
        Reconstruct symmetric distance matrix from upper-triangular format.

        Args:
            upper_triangular_distances: 1D array of shape [n_basis*(n_basis-1)/2]
            n_basis: Number of basis points

        Returns:
            Symmetric distance matrix of shape [n_basis, n_basis]
        """
        weights_matrix = np.zeros((n_basis, n_basis))
        idx = 0
        for i in range(n_basis):
            for j in range(i + 1, n_basis):
                weights_matrix[i, j] = upper_triangular_distances[idx]
                weights_matrix[j, i] = upper_triangular_distances[idx]
                idx += 1
        return weights_matrix

    def __init__(self,
                 source_data: np.ndarray,
                 basis_indices: np.ndarray,
                 upper_triangular_distances: np.ndarray,
                 n_neighbors: int = 25,
                 method: str = 'ensemble_knn',
                 batch_size: int = 128,
                 device: str = None,
                 precomputed_base_projections: np.ndarray = None,
                 verbose: bool = True):
        """
        Initialize Projector.

        Args:
            source_data: all data points [N, features]
            basis_indices: indices of basis points in source_data
            upper_triangular_distances: Upper-triangular distances [n_basis*(n_basis-1)/2] (optional)
            n_neighbors: number of neighbors for Isomap and interpolation (default: 25)
            method: projection method ('krr', 'ensemble_knn', 'random_forest')
            batch_size: batch size for projection methods
            device: 'cuda', 'cpu', or None (auto)
            precomputed_base_projections: precomputed base projections [base_dim, proj_dim]
            verbose: print progress messages
        """

        n_basis = len(basis_indices)
        weights_matrix = self.reconstruct_distance_matrix(upper_triangular_distances, n_basis)
        self.source_data = source_data.astype(float)
        self.weights_matrix = weights_matrix
        self.basis_indices = basis_indices
        self.n_neighbors = n_neighbors
        self.method = method
        self.batch_size = batch_size
        self.verbose = verbose

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if precomputed_base_projections is not None:
            if self.verbose:
                print(f"Projector: Using precomputed base projections: {precomputed_base_projections.shape}")
            self.base_projections = precomputed_base_projections
        else:
            if self.verbose:
                print("Projector: Base projections not provided - will compute when needed")
            self.base_projections = None

        self.projection = None

    def compute_base_projections(self):
        """
        Compute base projections using Isomap on basis points.

        Returns:
            base_projections: coordinates of basis points in hidden space [base_dim, proj_dim]
        """
        if self.verbose:
            print("Projector: Computing base projections with Isomap...")

        proj_dim = len(self.basis_indices)

        weights_tensor = torch.tensor(self.weights_matrix, dtype=torch.float64).to(self.device)

        isomap = IsomapNN(
            weights_initial_assumption=weights_tensor,
            n_components=proj_dim,
            n_neighbors=self.n_neighbors
        )

        projections = isomap.fit_transform(weights_tensor)
        self.base_projections = projections.detach().cpu().numpy()

        if self.verbose:
            print(f"Projector: Base projections computed: {self.base_projections.shape}")

        return self.base_projections

    def compute_projection(self):
        """
        Compute projections for ALL points using KNN interpolation from basis points.

        Returns:
            projection: coordinates of all points in hidden space [N, proj_dim]
        """
        # Ensure base projections exist
        if self.base_projections is None:
            self.compute_base_projections()

        if self.verbose:
            print(f"Projector: Computing projections for all {len(self.source_data)} points using {self.method}...")

        X_basis = self.source_data[self.basis_indices]
        Y_basis = self.base_projections
        X_all = self.source_data

        # Choose projection method
        if self.method == 'krr':
            Y_all = project_krr_optimized(X_basis, Y_basis, X_all, batch_size=self.batch_size)
        elif self.method == 'ensemble_knn':
            Y_all = project_ensemble_knn(X_basis, Y_basis, X_all)
        elif self.method == 'random_forest':
            Y_all = project_random_forest(X_basis, Y_basis, X_all, batch_size=self.batch_size)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        # Ensure basis points have exact projections (not interpolated)
        Y_all[self.basis_indices] = Y_basis

        self.projection = Y_all

        if self.verbose:
            print(f"Projector: All projections computed: {self.projection.shape}")

        return self.projection

    def save(self, folder_path: str):
        """
        Save both base and all projections to disk along with metadata.

        Args:
            folder_path: directory to save projections
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.base_projections is not None:
            base_path = os.path.join(folder_path, 'base_projections.npy')
            np.save(base_path, self.base_projections)
            if self.verbose:
                print(f"Projector: Saved base projections to {base_path}")

        if self.projection is not None:
            all_path = os.path.join(folder_path, 'projection.npy')
            np.save(all_path, self.projection)
            if self.verbose:
                print(f"Projector: Saved all projections to {all_path}")

        metadata = {
            'n_neighbors': self.n_neighbors,
            'method': self.method,
            'latent_dim': self.base_projections.shape[1] if self.base_projections is not None else None,
            'n_basis_points': len(self.basis_indices),
            'batch_size': self.batch_size,
            'device': self.device
        }
        metadata_path = os.path.join(folder_path, 'projector_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"Projector: Saved metadata to {metadata_path}")