import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraphRegularizer:
    """
    Класс для добавления графовой регуляризации к произвольной модели машинного обучения.

    Класс принимает координаты всех фичей в евкл пространстве (source_data),
    координаты базисных фичей в локальных координатах (proj_base_features),
    матрицу weights_matrix весов вычисленных по расстояниям по многообразию между базисными фичами,
    номера базисных фичей среди всех от 1 до N.
    Класс расширяет матрицу весов на все точки датасета, а потом вносит в лосс лапласиан от расширенной матрицы.
    """

    def __init__(self, weights_matrix, adjacency_matrix, basis_indices,
                   source_data, proj_base_features, lambda_graph=1.0, n_neighbors=5, method='knn'):
        self.W_basis = weights_matrix # [base_dim, base_dim]
        self.basis_indices = basis_indices
        self.adjacency_matrix = adjacency_matrix
        self.source_data = source_data #  [N, features]
        self.proj_base_features = proj_base_features
        self.lambda_graph = lambda_graph
        self.n_neighbors = n_neighbors
        self.method = method

        self.I = self._build_interpolation_matrix()

    def _build_interpolation_matrix(self, method='knn'):
        """
        Матрица перехода (интерполяции) между базисными и небазисными точками

        """
        N = self.source_data.shape[0]
        basis_size = len(self.basis_indices)
        I = np.zeros((N, basis_size))
        for basis_idx, point_idx in enumerate(self.basis_indices) :
            I[point_idx, basis_idx] = 1.0


        non_basis_indices = np.setdiff1d(np.arange(N), self.basis_indices)
        nbrs = NearestNeighbors(n_neighbours=self.n_neighbors, metric='euclidean').fit(self.source_data[self.basis_indices]) # подумать надо ли заменить на fit(self.roj[self.basis_indices])
        distances, neighbor_indices = nbrs.kneighbors(self.source_data[non_basis_indices])

        for i, point_idx in enumerate(non_basis_indices):

            weights = 1 / (distances[i, :] + 1e-8)
            weights = weights / np.sum(weights)
            I[point_idx, neighbor_indices[i]] = weights

        return I


    def graph_loss(self, predictions, batch_indices): #
        """
        Вычисляет графовый loss для батча предсказаний
        Тут предполагаю что получаем мы именно матрицу весов, а не расстояний

        """
        I_batch = self.I[batch_indices, :]
        W_batch = I_batch @ self.W_basis @ I_batch.T
        D_batch = np.diag(np.sum(W_batch, axis=1))
        L_batch = D_batch - W_batch

        part_1 = np.dot(predictions.T, L_batch)
        loss = np.dot(part_1, predictions)
        return loss.reshape(-1)[0]

    def combined_loss(self, model_loss, predictions, batch_indices):
        """
        Возвращает комбинированный loss = model_loss + lambda * graph_loss
        """
        g_loss = self.graph_loss(predictions, batch_indices)
        return model_loss + self.lambda_graph * g_loss
