import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class DataVisualizer:
    """Handles adaptive visualization for data of any dimensionality"""

    def __init__(self, original_features):
        self.original_features = original_features
        self.n_samples, self.n_features = original_features.shape
        self.dimensionality = min(3, self.n_features)
        self.use_pca = self.n_features > 3

        if self.use_pca:
            self.pca = PCA(n_components=3)
            self.pca.fit(original_features)
            self.explained_variance = self.pca.explained_variance_ratio_.sum()

    def project_data(self, features):
        """Project data to appropriate dimensionality"""
        if self.use_pca:
            return self.pca.transform(features)
        else:
            # Pad lower-dimensional data with zeros for consistent plotting
            if self.n_features == 1:
                return np.column_stack([features.flatten(), np.zeros_like(features.flatten())])
            elif self.n_features == 2:
                return np.column_stack([features, np.zeros(len(features))])
            else:  # 3 features exactly
                return features

    def create_scatter_plot(self, ax, data, colors, title, cmap='viridis'):
        """Create scatter plot adapted to data dimensionality"""
        if self.dimensionality == 3:
            sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                            c=colors, cmap=cmap, alpha=0.7, s=20)
            ax.set_zlabel('Component 3')
        elif self.dimensionality == 2:
            sc = ax.scatter(data[:, 0], data[:, 1],
                            c=colors, cmap=cmap, alpha=0.7, s=20)
        else:  # 1D
            sc = ax.scatter(data[:, 0], np.zeros_like(data[:, 0]),
                            c=colors, cmap=cmap, alpha=0.7, s=20)
            ax.set_yticks([])

        return sc

    def get_projection_type(self):
        """Get the appropriate projection type for subplots"""
        return '3d' if self.dimensionality == 3 else None

    def get_variance_info(self):
        """Get explained variance info (only relevant for PCA)"""
        if self.use_pca:
            return f"PCA Explained variance: {self.explained_variance:.3f}"
        else:
            return f"Original {self.n_features}D data"


def create_visualization(epoch, losses, best_epoch, best_loss, best_reproj_features,
                            best_outputs, reproj_features, output,
                            y_train, isomap_weights, isomap_eigenvalues, working_dir, current_time):
    """
    Create visualization for data using adaptive dimensionality handling
    """
    # Initialize visualizer with best features as reference
    visualizer = DataVisualizer(best_reproj_features)

    # Project all data consistently
    current_proj = visualizer.project_data(reproj_features)
    best_proj = visualizer.project_data(best_reproj_features)
    projection_type = visualizer.get_projection_type()
    variance_info = visualizer.get_variance_info()

    y_train = y_train.cpu().detach().numpy()
    fig = plt.figure(figsize=(18, 18))

    # 1. Convergence Plot
    ax1 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    ax1.plot(np.arange(len(losses)), losses, label='Train')
    ax1.axhline(best_loss, c='r', linestyle='dashed')
    ax1.annotate(str(round(best_loss, 4)), (0, best_loss), c='r')
    ax1.axvline(best_epoch, c='r', linestyle='dashed')
    ax1.annotate(best_epoch, (best_epoch, max(losses)), c='r')
    ax1.set_title(f'Convergence plot (Time spent: {current_time}), epoch={epoch}')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_yscale('log')
    ax1.legend()

    # 2. Weight distribution
    ax2 = plt.subplot2grid((3, 3), (0, 0))
    ax2.hist(isomap_weights, bins=50, color='skyblue', edgecolor='black')
    ax2.set_title(f'Distribution of isomap_weights,\neigenvalues={isomap_eigenvalues}')
    ax2.set_xlabel('Weight value')
    ax2.set_ylabel('Frequency')

    # 3. Best Features
    ax3 = plt.subplot2grid((3, 3), (2, 0), projection=projection_type)
    sc3 = visualizer.create_scatter_plot(ax3, best_proj, y_train, 'Best Features', 'plasma')
    ax3.set_title(f'Best Features - {variance_info}')
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    if visualizer.dimensionality >= 2:
        plt.colorbar(sc3, ax=ax3, shrink=0.6)

    # 4. Current Features
    ax4 = plt.subplot2grid((3, 3), (1, 0), projection=projection_type)
    sc4 = visualizer.create_scatter_plot(ax4, current_proj, y_train,
                                         f'Current Features (epoch {epoch})', 'viridis')
    ax4.set_title(f'Current Features (epoch {epoch})')
    ax4.set_xlabel('Component 1')
    ax4.set_ylabel('Component 2')
    if visualizer.dimensionality >= 2:
        plt.colorbar(sc4, ax=ax4, shrink=0.6)

    # 5. Current Predictions vs True Labels
    ax5 = plt.subplot2grid((3, 3), (1, 1))
    ax5.scatter(y_train, output.flatten(), alpha=0.6, s=20)
    ax5.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', alpha=0.8, label='Perfect prediction')
    ax5.set_xlabel('True Labels')
    ax5.set_ylabel('Model Outputs')
    ax5.set_title(f'Current Predictions vs True\nLoss: {losses[-1]:.5f}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Current Features colored by predictions
    ax6 = plt.subplot2grid((3, 3), (1, 2), projection=projection_type)
    sc6 = visualizer.create_scatter_plot(ax6, current_proj, output.flatten(),
                                         'Current Features colored by Predictions', 'viridis')
    ax6.set_title('Current Features colored by Predictions')
    ax6.set_xlabel('Component 1')
    ax6.set_ylabel('Component 2')
    if visualizer.dimensionality >= 2:
        plt.colorbar(sc6, ax=ax6, shrink=0.6)

    # 7. Best Predictions vs True Labels
    ax7 = plt.subplot2grid((3, 3), (2, 1))
    ax7.scatter(y_train, best_outputs.flatten(), alpha=0.6, s=20)
    ax7.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', alpha=0.8, label='Perfect prediction')
    ax7.set_xlabel('True Labels')
    ax7.set_ylabel('Model Outputs')
    ax7.set_title(f'Best Predictions vs True\nBest loss: {best_loss:.5f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Best Features colored by predictions
    ax8 = plt.subplot2grid((3, 3), (2, 2), projection=projection_type)
    sc8 = visualizer.create_scatter_plot(ax8, best_proj, best_outputs.flatten(),
                                         'Best Features colored by Predictions', 'plasma')
    ax8.set_title('Best Features colored by Predictions')
    ax8.set_xlabel('Component 1')
    ax8.set_ylabel('Component 2')
    if visualizer.dimensionality >= 2:
        plt.colorbar(sc8, ax=ax8, shrink=0.6)

    plt.tight_layout()
    plt.savefig(f'{working_dir}/{epoch}.png', dpi=150)
    plt.close()

