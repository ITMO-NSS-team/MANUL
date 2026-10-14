"""
Epoch-tracked diagnostics of the manifold structure that GradientIsomap's
free distance matrix W converges to.

Scope agreed with the user for this round (see
docs/regularization_investigation_journal.md, "epoch tracking" plan):

  Group A (global, every epoch - free, reuses KernelPCA's already-computed
  full spectrum): top-10 eigenvalues, lambda1/lambda2 ratio ("curvature"),
  spectral gap, effective rank, negative-mass fraction.

  Group C (local curvature only): a local, per-point curvature-like proxy
  from the local Gram-matrix spectrum of each point's k-NN submatrix,
  correlated epoch-by-epoch against analytic Gaussian curvature where a
  ground truth is available (torus).

  Group D (local structure, all): per-point neighbor-set stability across
  checkpoints (Jaccard) and a LOCAL (not global) triangle-inequality
  violation rate, since W has no constraint forcing the triangle
  inequality anywhere and violations are expected to be a real, meaningful
  property rather than a bug.

Per the corrected framing captured in memory
(feedback_manifold_diagnostics_framing): these are all INVARIANT
aggregate/local-structure properties, not raw value-by-value comparisons
across independent runs - the latter is not a meaningful signal-vs-noise
test for this underdetermined optimization.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Group A: global spectral diagnostics (cheap - reuses KernelPCA's full
# sorted spectrum, no extra eigendecomposition needed).
# ---------------------------------------------------------------------------

def global_spectral_diagnostics(full_eigs: np.ndarray, top_k: int = 10) -> dict:
    """
    full_eigs: 1D array, already sorted by |eigenvalue| descending (this is
    exactly isomap_model.kernel_pca_.full_eigenvalues_sorted_).
    """
    full_eigs = np.asarray(full_eigs, dtype=np.float64)
    top = full_eigs[:top_k]
    abs_full = np.abs(full_eigs)

    lambda1 = float(full_eigs[0]) if len(full_eigs) > 0 else float('nan')
    lambda2 = float(full_eigs[1]) if len(full_eigs) > 1 else float('nan')
    # Signed ratio, matching the user's "curvature-like" quantity: relation
    # between the two largest-by-magnitude eigenvalues. Sign is kept
    # deliberately (not abs) - a negative ratio means the two dominant
    # directions disagree on embeddability in real Euclidean space.
    lambda_ratio_12 = lambda1 / lambda2 if lambda2 != 0 else float('nan')

    spectral_gap = (abs(lambda1) - abs(lambda2)) / abs(lambda1) if lambda1 != 0 else float('nan')

    sum_abs = abs_full.sum()
    sum_sq = np.square(full_eigs).sum()
    # Participation-ratio style effective rank: (sum|lambda_i|)^2 / sum(lambda_i^2).
    # Unlike the classical entropy-based effective rank (Roy & Vetterli), this
    # is well-defined with negative eigenvalues (which the standard von Neumann
    # entropy formulation is not, since it needs lambda_i >= 0 to form a
    # probability distribution).
    effective_rank = float((sum_abs ** 2) / sum_sq) if sum_sq > 0 else float('nan')

    neg_mass = abs_full[full_eigs < 0].sum()
    neg_mass_frac = float(neg_mass / sum_abs) if sum_abs > 0 else float('nan')

    return {
        'top_eigenvalues': ','.join(f'{x:.6g}' for x in top),
        'lambda1': lambda1,
        'lambda2': lambda2,
        'lambda_ratio_12': float(lambda_ratio_12),
        'spectral_gap': float(spectral_gap),
        'effective_rank': effective_rank,
        'neg_mass_frac': neg_mass_frac,
    }


# ---------------------------------------------------------------------------
# Shared helper: k-NN indices from a (possibly non-metric) distance matrix.
# ---------------------------------------------------------------------------

def knn_indices_from_distance_matrix(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """Returns (N, k) int array of each point's k nearest neighbor indices
    (self excluded), using the distance matrix as given - no assumption
    that it obeys the triangle inequality."""
    n = distance_matrix.shape[0]
    k = min(k, n - 1)
    order = np.argsort(distance_matrix, axis=1)
    # first column is the point itself (distance 0 to itself, assuming a
    # proper diagonal); drop it defensively by filtering rather than
    # assuming column 0 is always self.
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = order[i]
        row = row[row != i]
        out[i] = row[:k]
    return out


# ---------------------------------------------------------------------------
# Group C: local curvature-like proxy via local Gram-matrix eigendecomposition.
# ---------------------------------------------------------------------------

def _local_gram_eigenvalues(local_dist: np.ndarray) -> np.ndarray:
    """Classical-MDS double-centering on a local k x k distance submatrix,
    returns eigenvalues sorted by |eigenvalue| descending."""
    k = local_dist.shape[0]
    d2 = np.square(local_dist)
    j = np.eye(k) - np.ones((k, k)) / k
    b = -0.5 * j @ d2 @ j
    b = (b + b.T) / 2.0  # symmetrize against float round-off
    eigvals = np.linalg.eigvalsh(b)
    order = np.argsort(-np.abs(eigvals))
    return eigvals[order]


def local_curvature_proxies(distance_matrix: np.ndarray, knn_indices: np.ndarray) -> dict:
    """
    For each point, build the local Gram matrix of {point} U {its k
    neighbors} from the (possibly non-metric) global distance matrix, and
    extract a curvature-like signal from its local eigenvalue spectrum:
    local lambda1/lambda2 ratio and local neg_mass_frac. This reuses
    exactly the same Gram-eigendecomposition machinery as the global
    diagnostic, just applied to a local neighborhood instead of all N
    points - the point of comparability the user asked for.

    Returns per-point arrays (length N).
    """
    n = distance_matrix.shape[0]
    local_lambda_ratio = np.full(n, np.nan)
    local_neg_mass_frac = np.full(n, np.nan)
    local_lambda1 = np.full(n, np.nan)

    for i in range(n):
        idx = np.concatenate(([i], knn_indices[i]))
        local_dist = distance_matrix[np.ix_(idx, idx)]
        eigvals = _local_gram_eigenvalues(local_dist)
        if len(eigvals) < 2:
            continue
        l1, l2 = eigvals[0], eigvals[1]
        local_lambda1[i] = l1
        local_lambda_ratio[i] = l1 / l2 if l2 != 0 else np.nan
        abs_sum = np.abs(eigvals).sum()
        if abs_sum > 0:
            local_neg_mass_frac[i] = np.abs(eigvals[eigvals < 0]).sum() / abs_sum

    return {
        'local_lambda1': local_lambda1,
        'local_lambda_ratio': local_lambda_ratio,
        'local_neg_mass_frac': local_neg_mass_frac,
    }


def torus_analytic_curvature(u: np.ndarray, R: float = 3.0, r: float = 1.0) -> np.ndarray:
    """
    Gaussian curvature of a standard torus parametrized by tube angle u
    (the angle around the tube's own circular cross-section):
        K(u) = cos(u) / (r * (R + r*cos(u)))
    Positive on the outer equator (u=0), negative on the inner equator
    (u=pi), zero at the top/bottom (u=+-pi/2) - the one geometry in this
    project's synthetic set with spatially-VARYING curvature, making it
    the only useful case for a correlation test against a local proxy.
    """
    u = np.asarray(u, dtype=np.float64)
    return np.cos(u) / (r * (R + r * np.cos(u)))


def correlate_local_curvature(local_proxy: np.ndarray, analytic: np.ndarray) -> dict:
    """Pearson and Spearman correlation between the local curvature proxy
    and analytic ground truth, ignoring NaNs pairwise."""
    mask = np.isfinite(local_proxy) & np.isfinite(analytic)
    if mask.sum() < 3:
        return {'pearson_r': float('nan'), 'spearman_r': float('nan'), 'n_valid': int(mask.sum())}
    x = local_proxy[mask]
    y = analytic[mask]
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    spearman_r = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {'pearson_r': pearson_r, 'spearman_r': spearman_r, 'n_valid': int(mask.sum())}


# ---------------------------------------------------------------------------
# Group D: local structure - neighbor stability across checkpoints, and
# local triangle-inequality violation rate.
# ---------------------------------------------------------------------------

def neighbor_stability(knn_prev: np.ndarray, knn_curr: np.ndarray) -> dict:
    """Per-point Jaccard similarity of k-NN sets between two checkpoints.
    High instability is expected under the underdetermined-optimization
    framing (many equally-valid configurations); this tracks HOW MUCH the
    local neighborhood structure itself is still moving, not whether it
    matches some canonical answer."""
    n = knn_prev.shape[0]
    jaccard = np.empty(n)
    for i in range(n):
        a = set(knn_prev[i].tolist())
        b = set(knn_curr[i].tolist())
        union = len(a | b)
        jaccard[i] = len(a & b) / union if union > 0 else np.nan
    return {'mean_jaccard': float(np.nanmean(jaccard)), 'per_point_jaccard': jaccard}


def local_triangle_violation_rate(distance_matrix: np.ndarray, knn_indices: np.ndarray,
                                    max_triples_per_point: int = 20,
                                    rtol: float = 1e-4, rng: np.random.Generator = None) -> dict:
    """
    Fraction of sampled local triples (i, j, k) - j, k both among i's
    k nearest neighbors - that violate the triangle inequality
    d(j,k) > d(i,j) + d(i,k) (up to a small relative tolerance for float
    round-off). Restricted to LOCAL triples (not arbitrary global ones)
    because W is only ever asked to be locally coherent for any given
    point's neighborhood; a global violation rate would conflate
    deliberate long-range non-metric structure with genuinely broken
    local geometry.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, k = knn_indices.shape
    if k < 2:
        return {'violation_rate': float('nan'), 'n_triples': 0}
    n_violations = 0
    n_triples = 0
    for i in range(n):
        neighbors = knn_indices[i]
        n_pairs_available = k * (k - 1) // 2
        n_triples_i = min(max_triples_per_point, n_pairs_available)
        pair_idx = rng.choice(n_pairs_available, size=n_triples_i, replace=False)
        for p in pair_idx:
            # unrank p into (a, b) with a < b among range(k)
            a = 0
            remaining = p
            while remaining >= (k - 1 - a):
                remaining -= (k - 1 - a)
                a += 1
            b = a + 1 + remaining
            j, kk = neighbors[a], neighbors[b]
            d_ij = distance_matrix[i, j]
            d_ik = distance_matrix[i, kk]
            d_jk = distance_matrix[j, kk]
            if d_jk > (d_ij + d_ik) * (1 + rtol):
                n_violations += 1
            n_triples += 1
    return {
        'violation_rate': float(n_violations / n_triples) if n_triples > 0 else float('nan'),
        'n_triples': n_triples,
    }
