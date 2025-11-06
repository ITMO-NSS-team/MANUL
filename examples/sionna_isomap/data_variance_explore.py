import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import Isomap
from torch import nn, float32
from scipy.spatial.distance import pdist, squareform

from Adam.Isomap import IsomapNN
from utils.DimensionalityAnalyser import DimensionalityAnalyser
device = 'cuda'


def calculate_isomap_quality(original_distances, embedded_points):
    """
    Calculate quality metrics for Isomap embedding
    """
    # Convert to numpy
    if torch.is_tensor(original_distances):
        original_dist = original_distances.cpu().numpy()
    else:
        original_dist = original_distances

    if torch.is_tensor(embedded_points):
        embedded_pts = embedded_points.cpu().numpy()
    else:
        embedded_pts = embedded_points

    # Calculate distances in embedded space
    embedded_dist = squareform(pdist(embedded_pts, metric='euclidean'))

    # Use upper triangle to avoid duplicates and diagonal
    n = original_dist.shape[0]
    idx = np.triu_indices(n, k=1)  # upper triangle without diagonal

    orig_dists_flat = original_dist[idx]
    embed_dists_flat = embedded_dist[idx]

    # Remove any zeros or infs
    mask = (orig_dists_flat > 0) & np.isfinite(orig_dists_flat)
    orig_dists = orig_dists_flat[mask]
    embed_dists = embed_dists_flat[mask]

    # Normalize distances to [0,1] range for stable calculation
    orig_dists_norm = orig_dists / np.max(orig_dists)
    embed_dists_norm = embed_dists / np.max(embed_dists)

    # Calculate metrics
    # 1. Stress (normalized)
    stress = np.sqrt(np.sum((orig_dists_norm - embed_dists_norm) ** 2) /
                     np.sum(orig_dists_norm ** 2))

    # 2. Pearson correlation between distances
    correlation = np.corrcoef(orig_dists, embed_dists)[0, 1]

    # 3. Trustworthiness-like metric
    trustworthiness = max(0, correlation ** 2)  # R-squared like

    return {
        'stress': stress,
        'correlation': correlation,
        'trustworthiness': trustworthiness
    }


H = np.load('sionna_sample.npy')  # [64, 4, 256, 20, 4]
H_flat = H.reshape(64, 4, 256, -1)  # [64, 4, 256, 80]
# Используем вещественную и мнимую части вместо амплитуды и фазы
real_part = np.real(H_flat)  # [64, 4, 256, 80]
imag_part = np.imag(H_flat)  # [64, 4, 256, 80]
H_processed = np.stack([real_part, imag_part], axis=-1)  # [64, 4, 256, 80, 2]
H_processed = H_processed.reshape(64 * 4 * 256 * 2, 80).T

'''analyser = DimensionalityAnalyser(max_neighbors=79)
analyser.analyse_dimensions(
    H_processed,
    method='both',
    n_samples=79
)
analyser.plot_dimension_histograms(dataset_name="sionna")
_, _ = analyser.plot_variance_threshold_analysis(H_processed,
                                                 dataset_name="sionna",
                                                 n_samples=79)'''

train_features = torch.tensor(H_processed, dtype=torch.float32).to('cuda')

results = []

dist = torch.cdist(train_features, train_features).to(device)
#for latent_len in [3, 50, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 131072]:
for latent_len in [2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80]:

    isomap_model = IsomapNN(dist, n_components=latent_len, eigval_choice='MDS')
    isomap_model.to(device)
    reproj_features = isomap_model().to(float32).detach()

    #iso = Isomap(n_components=latent_len)
    #reproj_features = iso.fit_transform(H_processed)

    print(f"Original distances - min: {dist.min():.4f}, max: {dist.max():.4f}, mean: {dist.mean():.4f}")
    print(f"Any NaN/inf in distances: {torch.isnan(dist).any()}, {torch.isinf(dist).any()}")
    print(f"Embedding - min: {reproj_features.min():.4f}, max: {reproj_features.max():.4f}")
    print(f"Embedding shape: {reproj_features.shape}")

    # Оцениваем качество embedding
    metrics = calculate_isomap_quality(dist, reproj_features)

    results.append({
        'latent_dim': latent_len,
        'stress': metrics['stress'],
        'correlation': metrics['correlation'],
        'trustworthiness': metrics['trustworthiness']
    })

    print(f"Latent dim: {latent_len:6d} | "
          f"Stress: {metrics['stress']:.4f} | "
          f"Correlation: {metrics['correlation']:.4f} | "
          f"Trustworthiness: {metrics['trustworthiness']:.4f}")

# Визуализация
dims = [r['latent_dim'] for r in results]
stresses = [r['stress'] for r in results]
correlations = [r['correlation'] for r in results]
trusts = [r['trustworthiness'] for r in results]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.semilogx(dims, stresses, 'o-', markersize=6)
plt.xlabel('Latent Dimension')
plt.ylabel('Stress')
plt.title('Stress vs Dimension')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.semilogx(dims, correlations, 'o-', markersize=6, color='green')
plt.xlabel('Latent Dimension')
plt.ylabel('Correlation')
plt.title('Distance Correlation vs Dimension')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.semilogx(dims, trusts, 'o-', markersize=6, color='red')
plt.xlabel('Latent Dimension')
plt.ylabel('Trustworthiness')
plt.title('Trustworthiness vs Dimension')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()