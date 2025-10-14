import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import time
from typing import Tuple


class ProjectionMethods:
    """
    Static methods for point-to-point transformation from sparse to full datasets.
    All methods are class-level and stateless.
    """

    @staticmethod
    def get_available_methods() -> list:
        """Return list of available projection methods."""
        return ['krr_optimized', 'ensemble_knn', 'random_forest', 'distance_calibration']

    @staticmethod
    def get_method_description(method_name: str) -> str:
        """Get description of each projection method."""
        descriptions = {
            'krr_optimized': 'Kernel Ridge Regression with hyperparameter optimization',
            'ensemble_knn': 'Ensemble of K-Nearest Neighbors regressors',
            'random_forest': 'Random Forest regression per dimension'
        }
        return descriptions.get(method_name, 'Unknown method')

    @staticmethod
    def project_krr_optimized(X_sparse: np.ndarray, Y_sparse: np.ndarray,
                              X_all: np.ndarray, batch_size: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Optimized Kernel Ridge Regression projection.

        Args:
            X_sparse: Sparse input points (n_sparse, n_features)
            Y_sparse: Sparse embeddings (n_sparse, n_components)
            X_all: All points to project (n_all, n_features)
            batch_size: Batch size for prediction

        Returns:
            Tuple of (projected_points, projection_time)
        """
        print("  Using KRR projection...")
        start_time = time.time()

        # Hyperparameter tuning
        param_grid = {'alpha': [0.1, 1.0, 10.0], 'gamma': [0.01, 0.1, 1.0]}
        krr = KernelRidge(kernel='rbf')

        # Use subset for tuning if large
        if len(X_sparse) > 1000:
            subset_idx = np.random.choice(len(X_sparse), 1000, replace=False)
            X_tune, Y_tune = X_sparse[subset_idx], Y_sparse[subset_idx]
        else:
            X_tune, Y_tune = X_sparse, Y_sparse

        grid_search = GridSearchCV(krr, param_grid, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(X_tune, Y_tune)
        best_krr = grid_search.best_estimator_
        best_krr.fit(X_sparse, Y_sparse)

        # Predict in batches
        Y_projected = []
        for i in range(0, len(X_all), batch_size):
            batch = X_all[i:i + batch_size]
            Y_projected.append(best_krr.predict(batch))

        projection_time = time.time() - start_time
        print(f"  KRR projection completed in {projection_time:.2f}s")

        return np.concatenate(Y_projected, axis=0), projection_time

    @staticmethod
    def project_ensemble_knn(X_sparse: np.ndarray, Y_sparse: np.ndarray,
                             X_all: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ensemble KNN regression projection.

        Args:
            X_sparse: Sparse input points (n_sparse, n_features)
            Y_sparse: Sparse embeddings (n_sparse, n_components)
            X_all: All points to project (n_all, n_features)

        Returns:
            Tuple of (projected_points, projection_time)
        """
        print("  Using Ensemble KNN projection...")
        start_time = time.time()

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

        projection_time = time.time() - start_time
        print(f"  Ensemble KNN projection completed in {projection_time:.2f}s")

        return Y_projected, projection_time

    @staticmethod
    def project_random_forest(X_sparse: np.ndarray, Y_sparse: np.ndarray,
                              X_all: np.ndarray, batch_size: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Random Forest regression projection.

        Args:
            X_sparse: Sparse input points (n_sparse, n_features)
            Y_sparse: Sparse embeddings (n_sparse, n_components)
            X_all: All points to project (n_all, n_features)
            batch_size: Batch size for prediction

        Returns:
            Tuple of (projected_points, projection_time)
        """
        print("  Using Random Forest projection...")
        start_time = time.time()

        n_components = Y_sparse.shape[1]
        Y_projected = np.zeros((len(X_all), n_components))

        for dim in range(n_components):
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=None,
                min_samples_split=5, n_jobs=-1, random_state=42
            )
            rf.fit(X_sparse, Y_sparse[:, dim])

            # Predict in batches
            dim_pred = []
            for i in range(0, len(X_all), batch_size):
                batch = X_all[i:i + batch_size]
                dim_pred.append(rf.predict(batch))

            Y_projected[:, dim] = np.concatenate(dim_pred)

        projection_time = time.time() - start_time
        print(f"  Random Forest projection completed in {projection_time:.2f}s")

        return Y_projected, projection_time
