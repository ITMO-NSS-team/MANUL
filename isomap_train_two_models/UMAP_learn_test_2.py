import torch
import numpy as np

import torch

import torch

def smacof_single_pytorch(
    dissimilarities,
    n_components=2,
    init=None,
    max_iter=300,
    eps=1e-3,
    random_state=None,
    device='cpu',
    verbose=0
):
    D = dissimilarities.to(device).double()  # Force double precision
    n_samples = D.shape[0]
    
    if random_state is not None:
        torch.manual_seed(random_state)
    
    # Initialize and scale X to match D's magnitude
    if init is None:
        X = torch.rand(n_samples, n_components, device=device, dtype=torch.double)
        with torch.no_grad():
            S_init = torch.cdist(X, X, p=2)
            scale = (D.mean() / S_init.mean()).item()
            X *= scale
    else:
        X = init.to(device).double()
    
    old_stress = None
    for it in range(max_iter):
        S = torch.cdist(X, X, p=2) + 1e-6  # Add epsilon to avoid division by zero
        stress = 0.5 * torch.sum((S - D)**2)
        
        # Guttman transform
        ratio = D / S
        B = -ratio
        diag_updates = ratio.sum(dim=1)
        B.diagonal().add_(diag_updates)
        
        X_new = (1.0 / n_samples) * (B @ X)
        
        # Update X and compute normalization
        X = X_new
        dis = torch.norm(X, dim=1).sum()
        if dis < 1e-6:
            dis = torch.tensor(1e-6, device=device)
        
        normalized_stress = stress / dis
        if verbose >= 2:
            print(f"Iter {it}: Stress = {stress.item()}")
        
        # Check convergence
        if old_stress is not None and (old_stress - normalized_stress) < eps:
            if verbose:
                print(f"Converged at iteration {it}")
            break
        old_stress = normalized_stress
    
    return X, stress.item(), it + 1
# Example usage:

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

if __name__ == '__main__':


    # Generate sample data
    X = np.random.rand(10, 5)
    D = pairwise_distances(X, metric='euclidean')

    # scikit-learn
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0,  eps=1e-10,  max_iter=1000)
    X_sklearn_init = mds.fit_transform(D)  # Get initial state
    mds.set_params(max_iter=300)
    X_sklearn = mds.fit_transform(D)
    print("scikit-learn Stress:", mds.stress_)

    # PyTorch
    D_tensor = torch.from_numpy(D).double()
    X_pytorch, stress, n_iter = smacof_single_pytorch(
        D_tensor,
        eps=1e-10,  # Disable early stopping
        max_iter=1000,
        random_state = 0
    )
    print("PyTorch Stress:", stress)