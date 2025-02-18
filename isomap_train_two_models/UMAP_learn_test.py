import torch
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes

def smacof_pytorch_optimized(
    dissimilarities,
    n_components=2,
    init=None,
    max_iter=300,
    eps=1e-6,
    verbose=0,
    normalized_stress=False,
    lr=0.1,
    device='cpu'
):
    if not torch.is_tensor(dissimilarities):
        D = torch.tensor(dissimilarities, dtype=torch.float32, device=device)
    else:
        D = dissimilarities.to(device).float()
    
    if not torch.allclose(D, D.t(), atol=1e-5):
        raise ValueError("The dissimilarity matrix must be symmetric.")
    
    n_samples = D.shape[0]
    
    if init is None:
        X = torch.randn(n_samples, n_components, device=device) * 0.01  # Small Gaussian noise
    else:
        X = torch.tensor(init, dtype=torch.float32, device=device)
        if X.shape[0] != n_samples or X.shape[1] != n_components:
            raise ValueError(f"init matrix should be of shape ({n_samples}, {n_components})")
    
    X.requires_grad_(True)  # Enable gradient tracking

    optimizer = torch.optim.Adam([X], lr=lr)  # Use Adam for better stability

    old_stress = None

    for it in range(max_iter):
        optimizer.zero_grad()

        # Compute pairwise Euclidean distances
        sum_X = torch.sum(X**2, dim=1, keepdim=True)
        dis_sq = sum_X + sum_X.t() - 2 * (X @ X.t())
        dis_sq = torch.clamp(dis_sq, min=0.0)
        dis = torch.sqrt(dis_sq + 1e-12)

        disparities = D  # Metric MDS

        # Compute stress
        stress = torch.sum((dis - disparities) ** 2) / 2
        if normalized_stress:
            norm_factor = torch.sum(disparities**2) / 2
            stress = torch.sqrt(stress / norm_factor)

        if verbose >= 2:
            print(f"Iteration {it}: stress = {stress.item()}")

        # Check convergence
        if old_stress is not None and abs(old_stress - stress.item()) < eps:
            if verbose:
                print(f"Converged at iteration {it} with stress {stress.item()}")
            break
        
        old_stress = stress.item()

        # Backpropagate and optimize
        stress.backward()
        optimizer.step()

    return X.cpu().detach().numpy(), stress.item(), it + 1


# Generate synthetic data
np.random.seed(42)
n_samples = 50
n_components = 2

X_true = np.random.rand(n_samples, n_components) * 10
D_true = squareform(pdist(X_true, metric="euclidean"))

noise = np.random.rand(n_samples, n_samples) * 0.1
D_noisy = D_true + (noise + noise.T) / 2  # Force symmetry

# Run scikit-learn's MDS
mds = MDS(n_components=n_components, dissimilarity="precomputed", max_iter=300, eps=1e-6, n_init=1)
X_sklearn = mds.fit_transform(D_noisy)
stress_sklearn = mds.stress_

# Run PyTorch SMACOF
X_pytorch, stress_pytorch, n_iter_pytorch = smacof_pytorch_optimized(D_noisy, n_components=n_components, max_iter=300, verbose=2, lr=0.1)

# Align embeddings using Procrustes
R, _ = orthogonal_procrustes(X_pytorch, X_sklearn)
X_pytorch_aligned = X_pytorch @ R

# Compute mean squared error between embeddings
embedding_error = np.mean((X_pytorch_aligned - X_sklearn) ** 2)

# Print results
print(f"Scikit-learn stress: {stress_sklearn}")
print(f"PyTorch stress: {stress_pytorch}")
print(f"Iterations in PyTorch: {n_iter_pytorch}")
print(f"Mean squared error between aligned embeddings: {embedding_error}")
