"""
A priori Euclidean-geometry baseline: pretrained Euclidean item embeddings
(classical MDS), no bilevel loop - the Euclidean geometry is fitted once,
ahead of time, then frozen for downstream NCF training. This is the direct
Euclidean counterpart of poincare_prior_baseline.py, completing the
three-way comparison:

  (1) Euclidean-pretrained NCF (this script, classical MDS)
  (2) GradientIsomapNCF (data-driven geometry search, run_eta_outer_sweep.py)
  (3) Poincare-pretrained NCF (a priori hyperbolic geometry)

Design mirrors poincare_prior_baseline.py line-for-line where possible, so
the ONLY thing that differs between (1) and (3) is which geometry the
pretrained item embeddings live in:

  - Euclidean embeddings z_i are fit via classical (Torgerson) MDS to
    minimize Euclidean stress against the SAME target used to initialize
    GradientIsomapNCF's D_input (Euclidean distances between item rating-
    profile vectors). Same target D_target, same input information as both
    other arms - only the geometry used to represent it differs.
  - Downstream: item embeddings are frozen, used directly (no tangent-space
    mapping needed - they are already Euclidean), then trained through the
    exact same NeuMFOnManifold head and ablation_geometry_vs_optimization.
    train_and_eval_ncf_on_fixed_Z harness used for the other two arms -
    identical downstream architecture and training protocol across all three
    comparisons.
  - Also runs geometry_diagnostics.py's ORC/spectral/persistent-homology
    measures directly on the fitted Euclidean distance matrix, as a
    reference point for interpreting GradientIsomapNCF's own measured values
    (a Euclidean MDS fit should look plainly non-hyperbolic: ORC ~ 0/positive,
    larger Gromov delta).
"""
"""
A priori Euclidean-geometry baseline: pretrained Euclidean item embeddings
(classical MDS), frozen for downstream NCF training. Direct Euclidean
counterpart of poincare_prior_baseline.py - same D_input_init target, same
train_and_eval_ncf_on_fixed_Z harness, so the only thing that differs
between the two arms is which geometry the pretrained item embeddings use.
"""
"""
A priori Euclidean-geometry baseline for SASRec: pretrained Euclidean item
embeddings (classical MDS), frozen for downstream SASRec training. Direct
Euclidean counterpart of poincare_prior_baseline.py, but wired into the
SASRec harness from ablation_geometry_vs_optimization_sasrec.py so the three
arms are directly comparable:

  (1) Euclidean-pretrained SASRec (this script, classical MDS -> frozen Z)
  (2) GradientIsomapSASRec       (data-driven geometry search, main run)
  (3) Poincare-pretrained SASRec (a priori hyperbolic geometry)

Same D_input_init target, same build_data_sasrec, same SASRecOnManifold head,
same train_and_eval_sasrec_on_fixed_Z training protocol as (2) - the only
thing that differs between arms is which geometry the frozen item embeddings
live in.
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_geometry_vs_optimization_sasrec import (
    build_data_sasrec, train_and_eval_sasrec_on_fixed_Z,
)


def fit_euclidean_embeddings(D_target, dim: int, device, seed: int = 0):
    """Classical (Torgerson) MDS: fit points in R^dim to reproduce
    D_target's pairwise distances. Deterministic, no random init - the
    classical MDS solution is unique up to sign flips / rotations, and the
    downstream SASRec head is invariant to both in expectation.

    Returns
    -------
    z            : [n, dim] float32 tensor on `device` (frozen item embeddings)
    d_final_flat : [n(n-1)/2] float64 numpy array of pairwise Euclidean
                   distances of the fitted points, matching the flat
                   upper-triangular convention used elsewhere in the pipeline.
    (iu, ju)     : triu_indices(n, n, offset=1) - index tensors for D_dense.
    n            : number of items.
    """
    torch.manual_seed(seed)
    D = D_target.detach().cpu().numpy().astype(np.float64) \
        if torch.is_tensor(D_target) else np.asarray(D_target, dtype=np.float64)
    n = D.shape[0]

    # Double centering: B = -0.5 * J D^2 J,  J = I - 1/n 11^T
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = 0.5 * (B + B.T)  # symmetrize away roundoff asymmetry

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Keep top-`dim` non-negative eigenvalues; clip tiny negatives from roundoff.
    top_vals = np.clip(eigvals[:dim], a_min=0.0, a_max=None)
    Z = eigvecs[:, :dim] * np.sqrt(top_vals)[None, :]
    z = torch.tensor(Z, dtype=torch.float32, device=device)

    # Reconstruct pairwise distances of the fit for the saved D matrix.
    iu, ju = torch.triu_indices(n, n, offset=1)
    d_full = torch.cdist(z, z, p=2)
    d_final_flat = d_full[iu.to(device), ju.to(device)].detach().cpu().numpy()

    return z.detach(), d_final_flat, (iu, ju), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon", choices=["movielens", "amazon"])
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--tag", default="",
                        help="Optional suffix for the saved geometry filename, "
                             "so runs at different scales don't overwrite each "
                             "other's euclidean_sasrec_geometry*.npz.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    sasrec_config = {'hidden_units': 64, 'maxlen': 50, 'num_blocks': 2,
                     'num_heads': 1, 'dropout_rate': 0.2}

    tmp_logs = os.path.join(HERE, "euclidean_tmp_logs")
    data = build_data_sasrec(args.max_users, args.max_movies, 5,
                             args.dataset_dir_name, device, tmp_logs,
                             args.dataset_type, args.amazon_category)
    print(f"num_users={data['num_users']} num_movies={data['num_movies']}", flush=True)

    # --- (1) Fit Euclidean embeddings to the same target D_input_init uses ---
    z_eucl, d_final_flat, (iu, ju), n = fit_euclidean_embeddings(
        data["D_input_init"], dim=args.latent_dim, device=device)
    print(f"Euclidean MDS done: n={n}, dim={args.latent_dim}", flush=True)

    # --- (2) Downstream: same SASRec-on-fixed-Z harness as the other arms ---
    print("\n--- training SASRec on Euclidean-pretrained (frozen MDS) embeddings ---", flush=True)
    hr, ndcg = train_and_eval_sasrec_on_fixed_Z(
        z_eucl, data, device, args.latent_dim, sasrec_config,
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        verbose=args.verbose)

    print(f"\nEuclidean-pretrained SASRec: test HR@10={hr:.4f} NDCG@10={ndcg:.4f}", flush=True)

    # --- (3) Save npz (D + metrics + Z) for the unified table ---
    D_dense = np.zeros((n, n), dtype=np.float64)
    D_dense[iu.numpy(), ju.numpy()] = d_final_flat
    D_dense[ju.numpy(), iu.numpy()] = d_final_flat
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(HERE, f"euclidean_sasrec_geometry{suffix}.npz")
    np.savez(out_path,
             D=D_dense,
             hr=hr, ndcg=ndcg,
             Z=z_eucl.cpu().numpy())
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
    