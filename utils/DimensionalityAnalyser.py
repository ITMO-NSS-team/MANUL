import numpy as np
import matplotlib.pyplot as plt
from .local_pca_implementation import local_pca_dimension


class DimensionalityAnalyser:
    """
    Comprehensive analyser for intrinsic dimensionality estimation with visualization.
    """

    def __init__(self, max_neighbors=None, default_threshold=0.95, default_curvature_threshold=0.2):
        self.max_neighbors = max_neighbors
        self.default_threshold = default_threshold
        self.default_curvature_threshold = default_curvature_threshold
        self.results = {}

    def analyse_dimensions(self, X, n_samples=500, method='both'):
        """
        Comprehensive dimensionality analysis with multiple methods.

        Parameters:
            X : array (n_samples, n_features)
                Input data
            n_samples : int
                Number of points to sample for analysis
            method : str ('eigenvalue', 'variance', 'both')
                Which method(s) to use
        """
        if self.max_neighbors is None:
            self.max_neighbors = min(X.shape[1] - 1, len(X) - 1)

        n_neighbors = min(self.max_neighbors, len(X) - 1)
        print(f"Using n_neighbors: {n_neighbors}")

        if method in ['eigenvalue', 'both']:
            print("Estimating dimensions using eigenvalue curvature method...")
            dims_curvature = local_pca_dimension(
                X, n_neighbors=n_neighbors, n_samples=n_samples,
                with_eigenvalues=True, curvature_threshold=self.default_curvature_threshold
            )
            self.results['eigenvalue'] = dims_curvature

        if method in ['variance', 'both']:
            print("Estimating dimensions using variance threshold method...")
            dims_variance = local_pca_dimension(
                X, n_neighbors=n_neighbors, n_samples=n_samples,
                with_eigenvalues=False, threshold=self.default_threshold
            )
            self.results['variance'] = dims_variance

    def plot_dimension_histograms(self, dataset_name: str, save_path: str = None):
        """
        Plot histograms of local dimension estimates.
        """
        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

        if n_methods == 1:
            axes = [axes]

        for idx, (method, dims) in enumerate(self.results.items()):
            ax = axes[idx]

            # Calculate statistics
            median_dim = np.median(dims)
            mean_dim = np.mean(dims)

            # Plot histogram
            ax.hist(dims, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(median_dim, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_dim:.1f}')
            ax.axvline(mean_dim, color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_dim:.1f}')

            ax.set_xlabel('Local Dimension')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{method.capitalize()} Method - {dataset_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        # Print summary statistics
        for method, dims in self.results.items():
            print(f"\n{method.upper()} Method Summary:")
            print(f"  Median dimension: {np.median(dims):.1f}")
            print(f"  Mean dimension: {np.mean(dims):.1f}")
            print(f"  Std dimension: {np.std(dims):.1f}")
            print(f"  Dimension range: {np.min(dims)} - {np.max(dims)}")
            unique_dims, counts = np.unique(dims, return_counts=True)
            print(f"  Most common dimension: {unique_dims[np.argmax(counts)]} (count: {np.max(counts)})")

    def plot_variance_threshold_analysis(self, X, dataset_name, thresholds=None, n_samples=500, save_path:str=None):
        """
        Analyze how dimension estimates change with different variance thresholds.
        """
        if thresholds is None:
            thresholds = np.arange(0.75, 1.0, 0.05)

        n_neighbors = self.max_neighbors

        dims_by_threshold = {}
        median_dims = []

        print("Calculating dimensions for different variance thresholds...")
        for thresh in thresholds:
            print(f"  Threshold {thresh:.2f}...")
            local_dims = local_pca_dimension(
                X, n_neighbors=n_neighbors, n_samples=n_samples,
                threshold=thresh, with_eigenvalues=True
            )
            dims_by_threshold[np.round(thresh, 2)] = local_dims
            median_dims.append(np.median(local_dims))

        # Plot 1: Median dimension vs threshold
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(thresholds, median_dims, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Variance Threshold')
        plt.ylabel('Median Local Dimension')
        plt.title(f'Median Dimension vs Variance Threshold\n{dataset_name}')
        plt.grid(True, alpha=0.3)

        # Plot 2: Cumulative distribution for each threshold
        plt.subplot(1, 2, 2)

        dim_at_095 = {}
        for thresh, dims in dims_by_threshold.items():
            dim_values, counts = np.unique(dims, return_counts=True)
            cumulative_counts = np.cumsum(counts) / np.sum(counts)
            plt.plot(dim_values, cumulative_counts, 'o-', label=f'Threshold = {thresh:.2f}', markersize=4)

            # Find dimension where 95% of points have this dimension or lower
            mask = cumulative_counts >= 0.95
            if np.any(mask):
                first_dim_above_095 = dim_values[mask][0]
                dim_at_095[thresh] = first_dim_above_095
            else:
                dim_at_095[thresh] = dim_values[-1]

        plt.axhline(y=0.95, color='r', linestyle='--', label='95% cumulative threshold')

        # Highlight the default threshold
        if self.default_threshold in dims_by_threshold:
            thresh = self.default_threshold
            dim_value = dim_at_095[thresh]
            plt.axvline(x=dim_value, color='r', linestyle='--',
                        label=f'Dim at thresh={thresh}: {dim_value}')

        plt.xlabel('Local Dimension')
        plt.ylabel('Cumulative Proportion')
        plt.title(f'Cumulative Distribution of Local Dimensions\n{dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        return dims_by_threshold, dim_at_095

    def get_latent_dim(self, method='eigenvalue'):
        """
        Get recommended latent dimension based on analysis.
        """
        if method not in self.results:
            raise ValueError(f"Method {method} not found in results. Available: {list(self.results.keys())}")

        dims = self.results[method]
        recommended_dim = int(np.mean(dims))

        print(f"Recommended latent dimension ({method} method): {recommended_dim}")
        return recommended_dim
