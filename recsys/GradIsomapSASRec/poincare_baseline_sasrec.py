"""
Poincaré baseline for SASRec: a priori hyperbolic geometry.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import geoopt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ablation_geometry_vs_optimization_sasrec import build_data_sasrec, train_and_eval_sasrec_on_fixed_Z

def fit_poincare_embeddings(D_target: torch.Tensor, dim: int, device, epochs=1000, lr=0.01, seed=0):
    """Гиперболический MDS: вписываем товары в шар Пуанкаре."""
    torch.manual_seed(seed)
    n = D_target.shape[0]
    manifold = geoopt.PoincareBall(c=1.0)

    init = torch.randn(n, dim, device=device) * 1e-3
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    iu, ju = torch.triu_indices(n, n, offset=1)
    target = D_target[iu, ju].to(device)
    target = target / target.max() * 3.0

    print(f"Fitting Poincare MDS: {n} points, dim={dim}...")
    for ep in range(epochs):
        optimizer.zero_grad()
        d_hyp = manifold.dist(z[iu], z[ju])
        loss = ((d_hyp - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if ep % 200 == 0: print(f"  epoch {ep} loss={loss.item():.6f}")

    with torch.no_grad():
        z_tangent = manifold.logmap0(z) # Перевод в касательное пространство для SASRec
    return z_tangent.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_users", type=int, default=300)
    parser.add_argument("--max_movies", type=int, default=800)
    parser.add_argument("--dataset_dir_name", default="amazon_beauty")
    parser.add_argument("--dataset_type", default="amazon")
    parser.add_argument("--amazon_category", default="Beauty_and_Personal_Care")
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sasrec_config = {'hidden_units': 64, 'maxlen': 50, 'num_blocks': 2, 'num_heads': 1, 'dropout_rate': 0.2}

    tmp_logs = os.path.join(HERE, "poincare_tmp_logs")
    data = build_data_sasrec(args.max_users, args.max_movies, 5, args.dataset_dir_name, device, tmp_logs, args.dataset_type, args.amazon_category)

    # 1. Фитим гиперболическую геометрию
    z_tangent = fit_poincare_embeddings(data["D_input_init"], dim=args.latent_dim, device=device)

    # 2. Учим SASRec на ней
    print("\n--- training SASRec on Poincare embeddings ---")
    hr, ndcg = train_and_eval_sasrec_on_fixed_Z(z_tangent, data, device, args.latent_dim, sasrec_config, verbose=True)
    
    print(f"\n=== POINCARE BASELINE SUMMARY ===")
    print(f"Downstream SASRec: HR@10={hr:.4f}  NDCG@10={ndcg:.4f}")

if __name__ == "__main__":
    main()
    