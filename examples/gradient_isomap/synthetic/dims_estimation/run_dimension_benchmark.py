"""
Benchmark for intrinsic-dimension estimators (the existing PCA-curvature
method used by DimensionalityAnalyser, vs the geometric MLE estimator)
against synthetic manifolds with EXACTLY known ground-truth dimension - both
the project's existing low-dimensional geometries (mostly d=1-2) and the
higher-dimensional generators in higher_dim_geometries.py (k-sphere,
k-flat-torus, k-affine subspace) spanning d up to 20, to see how each
estimator's bias behaves as true dimension grows, independent of what MNIST
itself gives.

Logs a CSV and a true-vs-estimated scatter plot into this folder.
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

from data.synthetic_geometries import geometries, noisy_manifold
from utils.local_pca_implementation import local_pca_dimension
from utils.intrinsic_dim_estimators import mle_dimension
from higher_dim_geometries import k_sphere, k_flat_torus, k_affine_subspace, add_noise

N_SAMPLES = 3000
NOISE_LEVELS = [0.0, 0.01]
PCA_CURVATURE_N_NEIGHBORS = 30  # a sane local neighborhood, NOT ambient_dim-1
PCA_SAMPLE_POINTS = 500  # points to evaluate per local-PCA call (matches DimensionalityAnalyser usage)


def evaluate(name, true_dim, X):
    pca_est = local_pca_dimension(
        X, n_neighbors=min(PCA_CURVATURE_N_NEIGHBORS, len(X) - 1),
        n_samples=min(PCA_SAMPLE_POINTS, len(X)),
        with_eigenvalues=True,
    )
    pca_est = float(np.percentile(pca_est, 90))
    mle_est = mle_dimension(X, k1=10, k2=20)
    return {
        'geometry': name,
        'true_dim': true_dim,
        'ambient_dim': X.shape[1],
        'pca_curvature_est': pca_est,
        'mle_est': mle_est,
    }


results = []

print("=" * 60)
print("Existing low-dimensional synthetic geometries (true dim ~1-2)")
print("=" * 60)
for name, (func, true_dim) in geometries.items():
    for noise in NOISE_LEVELS:
        X, _ = noisy_manifold(func, noise_percent=noise, n_samples=N_SAMPLES) if noise > 0 else func(N_SAMPLES)
        row = evaluate(f'{name} (noise={noise})', true_dim, X)
        row['noise'] = noise
        results.append(row)
        print(row)

print("=" * 60)
print("Higher-dimensional generators (true dim up to 20)")
print("=" * 60)
for k in (2, 5, 10, 20):
    X, _ = k_sphere(k, N_SAMPLES)
    for noise in NOISE_LEVELS:
        Xn = add_noise(X, noise)
        row = evaluate(f'sphere_{k}d (noise={noise})', k, Xn)
        row['noise'] = noise
        results.append(row)
        print(row)

for k in (2, 5, 10):
    X, _ = k_flat_torus(k, N_SAMPLES)
    for noise in NOISE_LEVELS:
        Xn = add_noise(X, noise)
        row = evaluate(f'flat_torus_{k}d (noise={noise})', k, Xn)
        row['noise'] = noise
        results.append(row)
        print(row)

for k in (2, 5, 10, 20):
    X, _ = k_affine_subspace(k, N_SAMPLES, ambient_dim=k * 4)
    for noise in NOISE_LEVELS:
        Xn = add_noise(X, noise)
        row = evaluate(f'affine_{k}d (noise={noise})', k, Xn)
        row['noise'] = noise
        results.append(row)
        print(row)

df = pd.DataFrame(results)
csv_path = os.path.join(OUT_DIR, 'dimension_benchmark_results.csv')
df.to_csv(csv_path, index=False)
print(f"\nSaved results to {csv_path}")

# --- true vs estimated scatter plot ---
fig, ax = plt.subplots(figsize=(8, 8))
max_dim = df['true_dim'].max()
ax.plot([0, max_dim], [0, max_dim], 'k--', alpha=0.5, label='perfect estimate')
ax.scatter(df['true_dim'], df['pca_curvature_est'], alpha=0.6, label='PCA curvature (current method)')
ax.scatter(df['true_dim'], df['mle_est'], alpha=0.6, label='MLE (Levina-Bickel)')
ax.set_xlabel('True dimension')
ax.set_ylabel('Estimated dimension')
ax.set_title('Intrinsic dimension estimators: true vs estimated')
ax.legend()
ax.grid(True, alpha=0.3)
plot_path = os.path.join(OUT_DIR, 'dimension_benchmark_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plot to {plot_path}")
