import numpy as np
from sklearn.neighbors import NearestNeighbors


def mle_dimension(X, k1=10, k2=20):
    """
    Levina-Bickel (2004) maximum likelihood intrinsic dimension estimator.

    Unlike PCA-variance-based methods, this looks purely at how the number of
    neighbors within a given radius grows (a Poisson-process model of nearest
    neighbor distances) - it targets the manifold's geometric/topological
    dimension, not how much variance is needed to reconstruct the data, so it
    is not inflated by noise spread across many low-variance directions.

    Args:
        X: (n_samples, n_features) data matrix
        k1, k2: range of neighborhood sizes to average the estimate over
                (Levina-Bickel recommend averaging over a range rather than a
                single k, since the estimate is itself somewhat k-dependent)

    Returns:
        float: estimated intrinsic dimension
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k2 + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # drop distance to self (always 0)

    m_hat_per_k = []
    for k in range(k1, k2 + 1):
        Tk = distances[:, k - 1]  # distance to k-th neighbor
        Tj = distances[:, :k - 1]  # distances to neighbors 1..k-1
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.log(Tk[:, None] / Tj)
        local_inv = np.nanmean(ratios, axis=1)
        local_dim = 1.0 / local_inv
        local_dim = local_dim[np.isfinite(local_dim)]
        m_hat_per_k.append(np.mean(local_dim))

    return float(np.mean(m_hat_per_k))
