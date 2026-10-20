"""
k-range stability scan for two dimension-estimation methods:
- the current PCA-curvature method (local_pca_dimension, 90th-percentile
  aggregate over many local neighborhoods - same architecture the project
  already uses, just scanned across neighborhood size k instead of a single
  fixed k)
- MLE (Levina-Bickel), evaluated at each single k (not its own internal
  k1..k2 averaging) so its own k-dependence is visible too

Run on both synthetic manifolds with KNOWN ground-truth dimension (to see
where each method's curve actually plateaus at the right answer) and on real
MNIST (no ground truth - the practical question we actually care about).

Logs a CSV and a per-dataset plot (dimension estimate vs k) into this folder.
"""
import os
import sys

REPO_ROOT = r"c:\Users\Julia\Documents\NSS_lab\MANUL"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from torchvision import datasets

from data.synthetic_geometries import sphere, swiss_roll
from higher_dim_geometries import k_sphere, k_affine_subspace
from utils.utils import split_data

K_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 783, 1000, 1500]


def scan_pca_curvature(X, eval_indices, k_values, curvature_threshold=0.2):
    """90th-percentile local-PCA-curvature estimate at each k, reusing one
    neighbor search (the expensive part for large datasets) across all k."""
    max_k = min(max(k_values), len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
    neighbor_idx = nbrs.kneighbors(X[eval_indices], return_distance=False)

    out = {}
    for k in k_values:
        k_eff = min(k, max_k)
        dims = []
        for row in neighbor_idx:
            idx = row[1:k_eff + 1]
            X_local = X[idx] - X[idx].mean(axis=0)
            evr = PCA().fit(X_local).explained_variance_ratio_
            dim = 0
            for j in range(len(evr)):
                if j > 0 and evr[j] < curvature_threshold * evr[j - 1]:
                    break
                dim += 1
            dims.append(dim)
        out[k] = float(np.percentile(dims, 90))
        print(f"    PCA-curvature k={k} (eff={k_eff}) -> {out[k]:.2f}")
    return out


def scan_mle(X, k_values):
    """Per-k MLE point estimate (not Levina-Bickel's own k1..k2 averaging),
    to see the method's own k-dependence, reusing one neighbor search."""
    max_k = min(max(k_values), len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]

    out = {}
    for k in k_values:
        k_eff = min(k, max_k)
        Tk = distances[:, k_eff - 1]
        Tj = distances[:, :k_eff - 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.log(Tk[:, None] / Tj)
        local_inv = np.nanmean(ratios, axis=1)
        local_dim = 1.0 / local_inv
        local_dim = local_dim[np.isfinite(local_dim)]
        out[k] = float(np.mean(local_dim))
        print(f"    MLE k={k} (eff={k_eff}) -> {out[k]:.2f}")
    return out


def run_dataset(name, X, true_dim, n_eval_pca, n_eval_mle, results):
    rng = np.random.RandomState(0)
    print(f"\n=== {name} (true_dim={true_dim}, ambient={X.shape[1]}, n={len(X)}) ===")

    eval_idx_pca = rng.choice(len(X), size=min(n_eval_pca, len(X)), replace=False)
    print("  PCA-curvature scan:")
    pca_curve = scan_pca_curvature(X, eval_idx_pca, K_VALUES)

    print("  MLE scan:")
    mle_curve = scan_mle(X, K_VALUES)

    for k in K_VALUES:
        results.append({
            'dataset': name, 'true_dim': true_dim, 'ambient_dim': X.shape[1],
            'k': k, 'pca_curvature_est': pca_curve[k], 'mle_est': mle_curve[k],
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_VALUES, [pca_curve[k] for k in K_VALUES], 'o-', label='PCA curvature (current method)')
    ax.plot(K_VALUES, [mle_curve[k] for k in K_VALUES], 's-', label='MLE (Levina-Bickel)')
    if true_dim is not None:
        ax.axhline(true_dim, color='k', linestyle='--', alpha=0.5, label=f'true dim = {true_dim}')
    ax.set_xscale('log')
    ax.set_xlabel('k (neighborhood size)')
    ax.set_ylabel('Estimated dimension')
    ax.set_title(f'Dimension estimate vs k - {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(OUT_DIR, f'k_scan_{name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {plot_path}")


results = []

print("#" * 60)
print("SYNTHETIC (known ground truth)")
print("#" * 60)

X, _ = sphere(3000)
run_dataset('sphere_2d', X, 2, n_eval_pca=300, n_eval_mle=300, results=results)

X, _ = swiss_roll(3000)
run_dataset('swiss_roll_2d', X, 2, n_eval_pca=300, n_eval_mle=300, results=results)

X, _ = k_sphere(10, 3000)
run_dataset('sphere_10d', X, 10, n_eval_pca=300, n_eval_mle=300, results=results)

X, _ = k_affine_subspace(10, 3000, ambient_dim=40)
run_dataset('affine_10d_flat', X, 10, n_eval_pca=300, n_eval_mle=300, results=results)

X, _ = k_sphere(20, 3000)
run_dataset('sphere_20d', X, 20, n_eval_pca=300, n_eval_mle=300, results=results)

print("\n" + "#" * 60)
print("REAL MNIST (no known ground truth)")
print("#" * 60)

mnist_dataset = datasets.MNIST(root=os.path.join(REPO_ROOT, 'data'), train=True, download=True)
X_all = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32) / 255.0
y_all = mnist_dataset.targets.numpy()
X_all = X_all[:60000]
y_all = y_all[:60000]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_all, y_all, (0.7, 0.15, 0.15))
print(f"MNIST X_train: {X_train.shape}")

run_dataset('mnist_train', X_train, None, n_eval_pca=80, n_eval_mle=300, results=results)

df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, 'k_stability_scan_results.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved full results to {csv_path}")
