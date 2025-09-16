import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def local_pca_dimension(X, n_neighbors=100, n_samples=None, threshold=0.95, with_eigenvalues=True,
                        curvature_threshold=0.2):
    """
    Estimate local intrinsic dimensionality using multiscale PCA.

    Parameters:
        X : array (n_samples, n_features)
            Input data matrix
        n_neighbors : int, default=100
            Number of neighbors for each local PCA
        n_samples : int or None, default=None
            Number of points to evaluate (None = use all)
        threshold : float, default=0.95
            Cumulative variance threshold when with_eigenvalues=False
        with_eigenvalues : bool, default=True
            If True, use eigenvalue curvature method; else use variance threshold
        curvature_threshold : float, default=0.2
            Eigenvalue drop threshold when with_eigenvalues=True

    Returns:
        dims : array (n_samples,)
            Local dimension estimates
    """
    if n_samples is None:
        sample_idx = np.arange(len(X))
    else:
        sample_idx = np.random.choice(len(X), size=n_samples, replace=False)

    dims = []
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)

    for i in sample_idx:
        neighborhood = nbrs.kneighbors([X[i]], return_distance=False)[0]
        X_local = X[neighborhood]
        X_local = X_local - X_local.mean(axis=0)
        pca = PCA().fit(X_local)

        if with_eigenvalues:
            evr = pca.explained_variance_ratio_
            dim = 0
            for k in range(len(evr)):
                if (k > 0) and (evr[k] < curvature_threshold * evr[k - 1]):
                    break
                dim += 1
            dims.append(dim)

        else:
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            dim = np.argmax(cum_var >= threshold) + 1
            dims.append(dim)

    return np.array(dims)
