import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from Adam.Isomap import IsomapNN


class Projector:
    """
    Class for computing and managing projections from Euclidean space to hidden geometry space.

    This class handles:
    1. Computing base projections via Isomap (or using precomputed ones)
    2. Computing projections for all points via KNN interpolation
    """

    def __init__(self,
                 source_data: np.ndarray,
                 base_indices: np.ndarray,
                 weights_matrix: np.ndarray = None,
                 base_projection: np.ndarray = None,
                 n_neighbors: int = 25,
                 batch_size: int = 128,
                 device: str = None):
        """
        Initialize Projector.

        Args:
            source_data: all data points [N, features]
            base_indices: indices of base points in source_data
            weights_matrix: weights matrix for base points to reproduce base_projection with Isomap
            base_projection: precomputed base projection [base_dim, proj_dim], reduce projection time if exists
            n_neighbors: number of neighbors for Isomap and interpolation (default: 25)
            batch_size: batch size for projection methods
            device: 'cuda', 'cpu', or None (auto)
        """

        self.source_data = source_data.astype(float)
        self.base_indices = base_indices
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._init_base_projection(base_projection, weights_matrix)

    @property
    def available_methods(self):
        return ['krr', 'ensemble_knn', 'random_forest']

    def _init_base_projection(self, base_projection, weights_matrix):
        if base_projection is None and weights_matrix is None:
            raise Exception('For entire points projection "base_projection" or "weights_matrix" must be '
                            'specified!')
        if base_projection is not None:
            self.base_projection = base_projection
        else:
            self.weights_matrix = weights_matrix
            self.base_projection = self.compute_base_projection()

    def compute_base_projection(self):
        """
        Compute base projections using Isomap on base points.
        Returns:
            base_projections: coordinates of base points in hidden space [base_dim, proj_dim]
        """
        print("Projector: Computing base projections with Isomap...")
        weights_tensor = torch.tensor(self.weights_matrix, dtype=torch.float64).to(self.device)
        isomap = IsomapNN(
            weights_initial_assumption=weights_tensor,
            n_components=len(self.base_indices),
            n_neighbors=self.n_neighbors)
        projections = isomap.fit_transform(weights_tensor)
        return projections.detach().cpu().numpy()

    def compute_projection(self, method='random_forest', save_base_distances=False):
        """
        Compute projections for all points from base points using selected interpolation method.
        Args:
            method: projection method ('krr', 'ensemble_knn', 'random_forest')
        Returns:
            projection: coordinates of all points in hidden space [N, proj_dim]
        """
        if method not in self.available_methods:
            raise Exception(f'Method "{method}" is not correct. Supported methods: {self.available_methods}')

        print(f"Projector: Computing projections for all {len(self.source_data)} points using {method}...")

        Y_base = self.base_projection
        Y_all = None

        if method == 'krr':
            Y_all = self.project_krr_optimized(batch_size=self.batch_size)
        elif method == 'ensemble_knn':
            Y_all = self.project_ensemble_knn()
        elif method == 'random_forest':
            Y_all = self.project_random_forest(batch_size=self.batch_size)

        if save_base_distances:
            # Ensure base points have exact projections (not interpolated)
            Y_all[self.base_indices] = Y_base
        return Y_all

    def project_krr_optimized(self, batch_size=1000):
        """Optimized Kernel Ridge Regression"""
        X_base = self.source_data[self.base_indices]
        Y_base = self.base_projection

        param_grid = {'alpha': [0.1, 1.0, 10.0], 'gamma': [0.01, 0.1, 1.0]}
        krr = KernelRidge(kernel='rbf')
        if len(X_base) > 1000:
            subset_idx = np.random.choice(len(X_base), 1000, replace=False)
            X_tune, Y_tune = X_base[subset_idx], Y_base[subset_idx]
        else:
            X_tune, Y_tune = X_base, Y_base
        grid_search = GridSearchCV(krr, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_tune, Y_tune)
        best_krr = grid_search.best_estimator_
        best_krr.fit(X_base, Y_base)
        Y_projected = []
        for i in range(0, len(self.source_data), batch_size):
            batch = self.source_data[i:i + batch_size]
            Y_projected.append(best_krr.predict(batch))
        return np.concatenate(Y_projected, axis=0)

    def project_ensemble_knn(self):
        """Ensemble KNN regression"""
        X_base = self.source_data[self.base_indices]
        Y_base = self.base_projection

        estimators = [
            ('knn5', KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)),
            ('knn10', KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)),
            ('knn15', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)),
        ]
        n_components = Y_base.shape[1]
        Y_projected = np.zeros((len(self.source_data), n_components))
        for dim in range(n_components):
            ensemble = VotingRegressor(estimators, n_jobs=-1)
            ensemble.fit(X_base, Y_base[:, dim])
            Y_projected[:, dim] = ensemble.predict(self.source_data)
        return Y_projected

    def project_random_forest(self, batch_size=1000):
        """Random Forest regression"""
        X_base = self.source_data[self.base_indices]
        Y_base = self.base_projection

        n_components = Y_base.shape[1]
        Y_projected = np.zeros((len(self.source_data), n_components))
        for dim in range(n_components):
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=None,
                min_samples_split=5, n_jobs=-1, random_state=42
            )
            rf.fit(X_base, Y_base[:, dim])
            dim_pred = []
            for i in range(0, len(self.source_data), batch_size):
                batch = self.source_data[i:i + batch_size]
                dim_pred.append(rf.predict(batch))
            Y_projected[:, dim] = np.concatenate(dim_pred)
        return Y_projected
