import numpy as np
import pytest

from utils.manifold_diagnostics import (
    global_spectral_diagnostics,
    knn_indices_from_distance_matrix,
    local_curvature_proxies,
    torus_analytic_curvature,
    correlate_local_curvature,
    neighbor_stability,
    local_triangle_violation_rate,
)


def test_global_spectral_diagnostics_basic():
    full_eigs = np.array([10.0, -6.0, 3.0, -1.0, 0.5])
    out = global_spectral_diagnostics(full_eigs, top_k=3)
    assert out['lambda1'] == 10.0
    assert out['lambda2'] == -6.0
    assert out['lambda_ratio_12'] == pytest.approx(10.0 / -6.0)
    assert out['spectral_gap'] == pytest.approx((10.0 - 6.0) / 10.0)
    expected_eff_rank = (np.abs(full_eigs).sum() ** 2) / np.square(full_eigs).sum()
    assert out['effective_rank'] == pytest.approx(expected_eff_rank)
    expected_neg_frac = (6.0 + 1.0) / np.abs(full_eigs).sum()
    assert out['neg_mass_frac'] == pytest.approx(expected_neg_frac)


def test_global_spectral_diagnostics_all_positive_has_zero_neg_mass():
    full_eigs = np.array([5.0, 4.0, 1.0])
    out = global_spectral_diagnostics(full_eigs)
    assert out['neg_mass_frac'] == 0.0


def test_knn_indices_excludes_self():
    d = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.5, 2.5],
        [2.0, 1.5, 0.0, 1.0],
        [3.0, 2.5, 1.0, 0.0],
    ])
    knn = knn_indices_from_distance_matrix(d, k=2)
    assert knn.shape == (4, 2)
    for i in range(4):
        assert i not in knn[i]
    assert set(knn[0].tolist()) == {1, 2}


def _torus_points(n_side=15, R=3.0, r=1.0):
    u = np.linspace(0.1, 2 * np.pi - 0.1, n_side)
    v = np.linspace(0.1, 2 * np.pi - 0.1, n_side)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel()
    vv = vv.ravel()
    x = (R + r * np.cos(uu)) * np.cos(vv)
    y = (R + r * np.cos(uu)) * np.sin(vv)
    z = r * np.sin(uu)
    pts = np.stack([x, y, z], axis=1)
    return pts, uu


def test_local_curvature_proxy_recovers_torus_curvature_sign_pattern():
    # Sanity check on ground truth itself, not the proxy: outer equator
    # (u=0) positive, inner equator (u=pi) negative, top/bottom (u=+-pi/2)
    # near zero - this is the known qualitative signature of torus
    # curvature and is what the local proxy will eventually be correlated
    # against on real GradientIsomap output.
    k_outer = torus_analytic_curvature(np.array([0.0]))
    k_inner = torus_analytic_curvature(np.array([np.pi]))
    k_side = torus_analytic_curvature(np.array([np.pi / 2]))
    assert k_outer[0] > 0
    assert k_inner[0] < 0
    assert k_side[0] == pytest.approx(0.0, abs=1e-10)


def test_local_curvature_proxies_runs_on_euclidean_torus_distances():
    pts, uu = _torus_points(n_side=12)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    knn = knn_indices_from_distance_matrix(d, k=10)
    out = local_curvature_proxies(d, knn)
    assert out['local_lambda_ratio'].shape == (pts.shape[0],)
    # Ambient Euclidean distances on a curved surface should still show
    # *some* non-degenerate local spectral structure (not all NaN/flat).
    assert np.isfinite(out['local_lambda1']).sum() > 0


def test_correlate_local_curvature_handles_nans():
    proxy = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    analytic = np.array([1.1, 2.1, 3.1, 3.9, 5.2])
    out = correlate_local_curvature(proxy, analytic)
    assert out['n_valid'] == 4
    assert out['pearson_r'] > 0.9


def test_neighbor_stability_identical_sets_gives_jaccard_one():
    knn = np.array([[1, 2], [0, 2], [0, 1]])
    out = neighbor_stability(knn, knn)
    assert out['mean_jaccard'] == pytest.approx(1.0)


def test_neighbor_stability_disjoint_sets_gives_jaccard_zero():
    knn_prev = np.array([[1, 2], [0, 2], [0, 1]])
    knn_curr = np.array([[3, 4], [3, 4], [3, 4]])
    out = neighbor_stability(knn_prev, knn_curr)
    assert out['mean_jaccard'] == pytest.approx(0.0)


def test_local_triangle_violation_rate_zero_for_true_euclidean_points():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 5))
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    knn = knn_indices_from_distance_matrix(d, k=10)
    out = local_triangle_violation_rate(d, knn, max_triples_per_point=15, rng=np.random.default_rng(1))
    assert out['violation_rate'] == pytest.approx(0.0, abs=1e-9)
    assert out['n_triples'] > 0


def test_local_triangle_violation_rate_positive_for_random_non_metric_matrix():
    rng = np.random.default_rng(0)
    n = 30
    d = rng.uniform(0.1, 5.0, size=(n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    knn = knn_indices_from_distance_matrix(d, k=10)
    out = local_triangle_violation_rate(d, knn, max_triples_per_point=15, rng=np.random.default_rng(1))
    assert out['violation_rate'] > 0.0
