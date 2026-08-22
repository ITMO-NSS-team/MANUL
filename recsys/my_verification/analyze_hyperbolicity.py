"""
Gromov delta-hyperbolicity analysis of the distance matrices produced by
GradientIsomapCF (recsys/GradIsomapCF_movielens/GradientIsomapCF_log.py).

Each outer epoch, GradientIsomapCF_log.save_epoch_matrices() saves a
matrices_epoch{k}.npz file with:
    D_input   - the learnable pairwise "input" distance matrix at that epoch
    D_geodesic- shortest-path (geodesic) distances over the kNN graph of D_input
    Z         - the manifold embedding (KernelPCA/MDS output) used by NCF
    D_latent  - pairwise Euclidean distances between rows of Z

We compute the exact 4-point (Gromov product) delta-hyperbolicity for each of
these matrices at each saved epoch, normalized by the matrix diameter
(delta_rel = 2*delta/diam), which is the standard scale-free way papers on
hyperbolic recommender systems report "how hyperbolic" a distance matrix is.
delta_rel = 0 for a tree metric (perfectly hyperbolic); delta_rel around
0.4-0.5 is typical for "generic" Euclidean point clouds / random graphs.

This lets us check the project's claim that the *learned* manifold (after the
outer-loop optimization) is closer to hyperbolic geometry than the raw,
un-optimized input space.
"""
import argparse
import glob
import os
import re

import numpy as np


def delta_hyperbolicity(D: np.ndarray, basepoint: int = 0) -> float:
    """Exact 4-point (Gromov product) delta-hyperbolicity of a distance matrix.

    Uses the standard basepoint trick (Fournier, Ismail & Vigneron 2015):
    with Gromov product (x|y)_w = 0.5*(d(w,x)+d(w,y)-d(x,y)), the
    hyperbolicity constant equals
        delta = max_{x,y,z} [ min((x|y)_w, (y|z)_w) - (x|z)_w ]
    independent of the choice of basepoint w. This reduces the naive O(n^4)
    four-point search to an O(n^3) max-min matrix "product", analogous to a
    single relaxation sweep of Floyd-Warshall but with (max, min) in place of
    (min, +).
    """
    n = D.shape[0]
    w = basepoint
    # Gromov product matrix A[x,y] = (x|y)_w
    dw = D[w, :]  # d(w, .)
    A = 0.5 * (dw[:, None] + dw[None, :] - D)

    # running max-min "matrix power": M[x,z] = max_y min(A[x,y], A[y,z])
    M = np.full((n, n), -np.inf, dtype=D.dtype)
    for y in range(n):
        cand = np.minimum(A[:, y:y + 1], A[y:y + 1, :])
        np.maximum(M, cand, out=M)

    delta = float(np.max(M - A))
    return delta


def summarize_matrix(name, D):
    n = D.shape[0]
    diam = float(D.max())
    delta = delta_hyperbolicity(D, basepoint=0)
    delta_rel = (2.0 * delta / diam) if diam > 0 else float("nan")
    print(f"  {name:12s} n={n:4d}  diam={diam:8.4f}  delta={delta:8.4f}  delta_rel(2*delta/diam)={delta_rel:.4f}")
    return {"name": name, "n": n, "diam": diam, "delta": delta, "delta_rel": delta_rel}


def analyze_run(logs_folder: str, max_epochs_to_show: int = None):
    print(f"\n=== Analyzing {logs_folder} ===")

    results = []

    init_path = os.path.join(logs_folder, "D_input_init.npy")
    if os.path.exists(init_path):
        D_init = np.load(init_path).astype(np.float64)
        print("\n[Initial D_input, before any outer-loop optimization]")
        results.append(("init", summarize_matrix("D_input_0", D_init)))

    epoch_files = glob.glob(os.path.join(logs_folder, "matrices_epoch*.npz"))

    def epoch_num(path):
        m = re.search(r"matrices_epoch(\d+)\.npz", os.path.basename(path))
        return int(m.group(1)) if m else -1

    epoch_files = sorted(epoch_files, key=epoch_num)
    if max_epochs_to_show is not None:
        shown = sorted(set([epoch_num(p) for p in epoch_files]))
        keep = set(shown[:1] + shown[-max_epochs_to_show:]) if len(shown) > max_epochs_to_show else set(shown)
        epoch_files = [p for p in epoch_files if epoch_num(p) in keep]

    for path in epoch_files:
        k = epoch_num(path)
        data = np.load(path)
        print(f"\n[Outer epoch {k}]")
        for key in ["D_input", "D_geodesic", "D_latent"]:
            if key in data:
                D = data[key].astype(np.float64)
                results.append((f"epoch{k}_{key}", summarize_matrix(key, D)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", required=True,
                         help="e.g. .../logs_movielens_isomap_cf/main01")
    parser.add_argument("--max_epochs_to_show", type=int, default=6,
                         help="show only first + last N epochs (for long runs)")
    args = parser.parse_args()
    analyze_run(args.logs_folder, args.max_epochs_to_show)


if __name__ == "__main__":
    main()
