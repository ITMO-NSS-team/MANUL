"""
A priori hyperbolic-geometry baseline: pretrained Poincare-ball item
embeddings (geoopt), no bilevel loop - the hyperbolic manifold is fitted
once, ahead of time, then frozen for downstream NCF training. This is the
missing third arm the paper needs for a real three-way comparison:

  (1) NeuMF (Euclidean baseline, this script's build_euclidean_baseline)
  (2) GradientIsomapNCF (data-driven geometry search, see run_eta_outer_sweep.py)
  (3) Poincare-pretrained NCF (a priori hyperbolic geometry, this script)

Without (3), "hyperbolic" in the paper only ever referred to a measured
characteristic of (2)'s own found geometry, never an independently-fitted
competing model - a real gap flagged during review.

Design:
  - Poincare embeddings z_i are fit via Riemannian gradient descent
    (geoopt.optim.RiemannianAdam) to minimize hyperbolic-distance stress
    against the SAME target used to initialize GradientIsomapNCF's D_input
    (Euclidean distances between item rating-profile vectors) - so both
    methods start from identical information, differing only in what
    geometry they're allowed/fit to use. This is a fair comparison, not
    "hyperbolic vs whatever GradientIsomapNCF happened to find".
  - Downstream: item embeddings are frozen, mapped to the tangent space at
    the origin (manifold.logmap0) so they're safe to feed into ordinary
    Euclidean linear layers (the standard hyperbolic-to-Euclidean interface
    trick), then trained through the exact same NeuMFOnManifold head and
    ablation_geometry_vs_optimization.train_and_eval_ncf_on_fixed_Z harness
    used for the other two arms - identical downstream architecture and
    training protocol across all three comparisons, only the input geometry
    differs.
  - Also runs geometry_diagnostics.py's own ORC/spectral/persistent-homology
    measures directly on the fitted Poincare distance matrix, as a sanity
    check that the diagnostic toolkit actually reports strongly hyperbolic
    characteristics (very negative ORC, small Gromov delta) for embeddings
    that ARE hyperbolic by construction - a reference point for interpreting
    GradientIsomapNCF's own measured values.
"""
import argparse
import os
import sys
import time

import geoopt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MANUL_DIR = os.path.dirname(os.path.dirname(HERE))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, MANUL_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from prepare_data import build_movie_user_matrix
from ablation_geometry_vs_optimization import build_data, train_and_eval_ncf_on_fixed_Z
from geometry_diagnostics import (
    ollivier_ricci_curvature, graph_laplacian_spectral, persistent_homology, _progress,
)
from analyze_hyperbolicity import delta_hyperbolicity


def fit_poincare_embeddings(D_target: torch.Tensor, dim: int, device, epochs: int = 1000,
                            lr: float = 0.01, seed: int = 0):
    """Hyperbolic MDS: fit Poincare-ball points to reproduce D_target's
    pairwise distances via manifold.dist(), analogous to what classical MDS
    does in Euclidean space - the direct hyperbolic-geometry counterpart of
    GradientIsomapNCF's Euclidean D_input initialization, so both methods
    start from the same distance information.
    """
    torch.manual_seed(seed)
    n = D_target.shape[0]
    manifold = geoopt.PoincareBall(c=1.0)

    init = torch.randn(n, dim, device=device) * 1e-3  # near origin, standard practice
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    iu, ju = torch.triu_indices(n, n, offset=1)
    target = D_target[iu, ju].to(device)
    # Scale target distances into a range Poincare distances can actually
    # reach without needing points to approach the ball boundary (where
    # gradients vanish) - same normalization spirit as D_input's [0,1] scaling.
    target = target / target.max() * 3.0

    _progress(f"Poincare MDS: fitting {n} points, dim={dim}, {epochs} epochs...")
    t0 = time.time()
    for ep in range(epochs):
        optimizer.zero_grad()
        d_hyp = manifold.dist(z[iu], z[ju])
        loss = ((d_hyp - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if ep % 200 == 0 or ep == epochs - 1:
            _progress(f"Poincare MDS: epoch {ep}/{epochs} loss={loss.item():.6f} "
                      f"({time.time() - t0:.1f}s elapsed)")

    with torch.no_grad():
        z_tangent = manifold.logmap0(z)  # safe Euclidean representation for downstream layers
        d_final = manifold.dist(z[iu], z[ju])

    return z.detach(), z_tangent.detach(), manifold, d_final.cpu().numpy(), (iu, ju), n


def evaluate_hyperbolicity_of_fit(d_pairwise_flat, iu, ju, n):
    """Reconstructs the dense hyperbolic distance matrix and runs our own
    diagnostic toolkit on it - sanity check that ORC/delta correctly report
    strong hyperbolicity for a manifold that IS hyperbolic by construction."""
    D = np.zeros((n, n), dtype=np.float64)
    D[iu, ju] = d_pairwise_flat
    D[ju, iu] = d_pairwise_flat

    delta = delta_hyperbolicity(D, basepoint=0)
    diam = float(D.max())
    delta_rel = (2.0 * delta / diam) if diam > 0 else float("nan")
    print(f"[Poincare geometry] Gromov delta={delta:.4f} delta_rel={delta_rel:.4f} "
          f"(0=perfectly tree-like/hyperbolic; MDS-derived Euclidean-init D_input in the "
          f"main sweep should be compared against this as a reference point)")

    # Build a kNN graph over the fitted hyperbolic distances for ORC/spectral
    # (same k=10 convention used elsewhere in this pipeline).
    k = 10
    knn_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(D[i])[1:k + 1]
        for j in neighbors:
            knn_adj[i, j] = D[i, j]
            knn_adj[j, i] = D[j, i]

    orc = ollivier_ricci_curvature(knn_adj)
    spec = graph_laplacian_spectral(knn_adj)
    print(f"[Poincare geometry] ORC mean={orc['mean']:.4f} f_neg={orc['f_neg']:.4f} "
          f"(reference: strongly hyperbolic should be clearly negative, f_neg close to 1)")
    print(f"[Poincare geometry] lambda2={spec['lambda2']:.4f} spectral_gap={spec['spectral_gap']:.4f}")

    ph = persistent_homology(D, max_edge_length=float(np.percentile(D[iu, ju], 3)))
    print(f"[Poincare geometry] H1 count={ph['h1_count']} H1 max persistence={ph['h1_max_persistence']:.4f}")

    return {"delta": delta, "delta_rel": delta_rel, "orc_mean": orc["mean"], "orc_f_neg": orc["f_neg"],
            "lambda2": spec["lambda2"], "spectral_gap": spec["spectral_gap"], "h1_count": ph["h1_count"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="ml-1m")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--mds_epochs", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tmp_logs_folder = os.path.join(HERE, "poincare_tmp_logs")
    data = build_data(args.max_users, args.max_movies, min_seq_len=2, num_ng=2,
                      dataset_dir_name=args.dataset_dir_name, device=device,
                      tmp_logs_folder=tmp_logs_folder)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    # --- (1) Fit Poincare embeddings to the same target D_input^(0) uses ---
    z_ball, z_tangent, manifold, d_final_flat, (iu, ju), n = fit_poincare_embeddings(
        data["D_input_init"], dim=args.latent_dim, device=device, epochs=args.mds_epochs)

    # --- (2) Sanity-check our diagnostics against a manifold that IS
    #     hyperbolic by construction ---
    print("\n--- geometry diagnostics on the fitted Poincare manifold (sanity check) ---", flush=True)
    hyp_diag = evaluate_hyperbolicity_of_fit(d_final_flat, iu.numpy(), ju.numpy(), n)

    # --- (3) Downstream: same NeuMFOnManifold head, same training harness,
    #     frozen Poincare (tangent-mapped) embeddings as the only input ---
    print("\n--- training final NCF on Poincare-pretrained embeddings ---", flush=True)
    hr, ndcg, val_loss = train_and_eval_ncf_on_fixed_Z(z_tangent, data, device, args.latent_dim)
    print(f"Poincare-pretrained NCF: test HR@10={hr:.4f} NDCG@10={ndcg:.4f} "
          f"best_val_loss={val_loss:.4f}", flush=True)

    print("\n=== POINCARE BASELINE SUMMARY ===")
    print(f"Fitted-manifold hyperbolicity: Gromov delta_rel={hyp_diag['delta_rel']:.4f}  "
          f"ORC mean={hyp_diag['orc_mean']:.4f}  f_neg={hyp_diag['orc_f_neg']:.4f}  "
          f"H1 count={hyp_diag['h1_count']}")
    print(f"Downstream: HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

    # Persist the fitted dense distance matrix + downstream metrics so
    # full_hyperbolicity_table.py can fold this arm into the unified table
    # without needing to refit (fitting is stochastic-ish across torch/CUDA
    # versions even with a fixed seed, so re-using the exact matrix that
    # produced the reported downstream numbers is more honest than refitting).
    D_dense = np.zeros((n, n), dtype=np.float64)
    D_dense[iu.numpy(), ju.numpy()] = d_final_flat
    D_dense[ju.numpy(), iu.numpy()] = d_final_flat
    out_path = os.path.join(HERE, "poincare_fitted_geometry.npz")
    np.savez(out_path, D=D_dense, val_loss=val_loss, hr=hr, ndcg=ndcg)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
