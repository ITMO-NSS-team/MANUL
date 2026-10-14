"""
Synthetic manifolds with EXACTLY known ground-truth intrinsic dimension,
spanning higher dimensions than the project's existing synthetic_geometries.py
(which is mostly true dim 1-2) - used to check how dimension estimators
behave as true dimension grows. All generators use genuinely random sampling
(not a regular grid), unlike synthetic_geometries.py's meshgrid-based points.
"""
import numpy as np


def k_sphere(k, n_samples, seed=0):
    """Uniform points on the k-sphere S^k embedded in R^(k+1). True dim = k."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, k + 1)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    colors = X[:, 0]
    return X, colors


def k_flat_torus(k, n_samples, seed=0):
    """Flat k-torus (S^1)^k, isometrically embedded in R^(2k). True dim = k."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, size=(n_samples, k))
    X = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    colors = angles[:, 0]
    return X, colors


def k_affine_subspace(k, n_samples, ambient_dim, seed=0):
    """Random k-dim linear subspace embedded in R^ambient_dim. True dim = k, zero curvature."""
    rng = np.random.RandomState(seed)
    coords = rng.uniform(-1, 1, size=(n_samples, k))
    q, _ = np.linalg.qr(rng.randn(ambient_dim, k))
    X = coords @ q.T
    colors = coords[:, 0]
    return X, colors


def add_noise(X, noise_percent, seed=0):
    if noise_percent == 0.0:
        return X
    rng = np.random.RandomState(seed)
    return X + rng.normal(0, np.max(np.abs(X)) * noise_percent, X.shape)
