from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt

from Eva.PopulationEvoOperators import IndividEvoOperators
from structure_approximation.IntrinsicNN import IntrinsicNN


class Population:
    def __init__(self, size: int, base_individ):
        """
        Class for population generation from a single individ by repeatedly applying mutation operator.
        :param size: int - number of individs in population
        :param base_individ: DataStructureGraph with individ to mutate
        """
        self.size = size
        self.base_individ = base_individ
        self.individs_pool = []

    def generate(self, edges_mutation: bool = True,
                 edges_weight_mutation: bool = True):
        """
        Function to generate population from the base individ to pool by mutation.
        :param edges_mutation: add and remove edges
        :param edges_weight_mutation: change edges length
        :return: class object with individs pool
        """

        individs_pool = [self.base_individ]
        while len(individs_pool) < self.size:
            new_indvid = deepcopy(self.base_individ)
            mutator = IndividEvoOperators([new_indvid],
                                          edges_mutation,
                                          edges_weight_mutation)
            new_indvid = mutator.mutate()[0]
            individs_pool.append(new_indvid)
            # filter invalid eigenvalues
            '''if new_indvid.valid_eigenvalues:
                new_indvid.visualize()
                individs_pool.append(new_indvid)
                print(f'Generate individ {len(individs_pool)} / {self.size}')'''

        self.individs_pool = individs_pool
        return self

    def visualize_population(self, save_path: str = None, figsize_per_plot=(4, 3), max_cols=4):
        """
        Visualize all individuals in the population in a grid layout

        :param save_path: Path to save the figure
        :param figsize_per_plot: Base figure size for each subplot
        :param max_cols: Maximum number of columns in the grid
        """
        self.individs_pool.sort(key=lambda x: x.fitness)
        n_individs = len(self.individs_pool)
        if n_individs == 0:
            print("No individuals in population to visualize")
            return

        n_cols = min(max_cols, n_individs)
        n_rows = (n_individs + n_cols - 1) // n_cols

        fig_width = n_cols * figsize_per_plot[0]
        fig_height = n_rows * figsize_per_plot[1]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        if n_individs == 1:
            axes = np.array([axes])
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()
        for idx, (individ, ax) in enumerate(zip(self.individs_pool, axes_flat)):
            if individ.features.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(individ.features)
                explained_var = pca.explained_variance_ratio_.sum()
                projection_info = f"\n(PCA: {explained_var:.3f} var)"
            else:
                features_2d = individ.features
                projection_info = ''
            ax.scatter(features_2d[:, 0], features_2d[:, 1],
                       s=5, alpha=0.7, c=individ.targets)

            n_nodes = individ.number_of_nodes
            max_possible_edges = n_nodes * (n_nodes - 1) / 2
            actual_edges = individ.number_of_edges

            if max_possible_edges != actual_edges:
                rows, cols = np.where(np.triu(individ.distances_matrix) > 0)
                if len(rows) > 0:
                    from matplotlib.collections import LineCollection
                    segments = np.array([[features_2d[i], features_2d[j]] for i, j in zip(rows, cols)])
                    lc = LineCollection(segments, colors='gray', alpha=0.1, linewidths=0.5)
                    ax.add_collection(lc)

            fitness_info = f", fit: {individ.fitness:.3f}" if individ.fitness is not None else ""
            ax.set_title(
                f'Ind {idx}: {individ.number_of_nodes} nodes, {individ.number_of_edges} edges, {fitness_info}{projection_info}',
                fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

        for idx in range(len(self.individs_pool), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def evaluate_individs_fitness(self):
        """
        Function for adding fitness parameter to each individ of the population.
        """
        for individ in self.individs_pool:
            if individ.fitness is None:
                features = torch.tensor(individ.features)
                targets = torch.tensor(individ.targets)
                dims = individ.dimensionality
                individ_model = IntrinsicNN(features,
                                            targets,
                                            dims,
                                            plot_convergence=False,
                                            epochs=500)

                individ_model.train()
                individ.loss = individ_model.loss
                fitness = 1 / individ.loss
                individ.fitness = fitness

        return self
