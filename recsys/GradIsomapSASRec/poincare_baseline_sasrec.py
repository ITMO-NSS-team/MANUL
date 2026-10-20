"""
Poincaré baseline for SASRec: a priori hyperbolic geometry.
Сохраняет геометрию в npz для последующего анализа delta_rel.
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
from ablation_hypersasrec import train_and_eval_hypsasrec_on_fixed_Z_nolog

def fit_poincare_embeddings(D_target: torch.Tensor, dim: int, device, epochs=1000, lr=0.01, seed=0):
    """Гиперболический MDS: вписываем товары в шар Пуанкаре."""
    torch.manual_seed(seed)
    n = D_target.shape[0]

    c = 0.1

    manifold = geoopt.PoincareBall(c)

    init = torch.randn(n, dim, device=device) * 1e-3
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    iu, ju = torch.triu_indices(n, n, offset=1)
    target = D_target[iu, ju].to(device)
    #target = target / target.max() * 3.0
    target = target / target.max() * (3.0 / (c ** 0.5))

    print(f"Fitting Poincare MDS: {n} points, dim={dim}...")
    for ep in range(epochs):
        optimizer.zero_grad()
        d_hyp = manifold.dist(z[iu], z[ju])
        loss = ((d_hyp - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if ep % 200 == 0: print(f"  epoch {ep} loss={loss.item():.6f}")

    with torch.no_grad():
        # Считаем финальные гиперболические расстояния для сохранения
        d_final = manifold.dist(z[iu], z[ju]).cpu().numpy()
        # Перевод в касательное пространство для SASRec
        z_tangent = manifold.logmap0(z).detach().cpu().numpy() 
        
    return z_tangent, d_final, (iu.cpu().numpy(), ju.cpu().numpy()), n

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
    z_tangent_np, d_final_flat, (iu, ju), n = fit_poincare_embeddings(
        data["D_input_init"].cpu(), dim=args.latent_dim, device=device
    )

    # 2. Сохраняем геометрию (матрицу гиперболических расстояний)
    D_dense = np.zeros((n, n), dtype=np.float64)
    D_dense[iu, ju] = d_final_flat
    D_dense[ju, iu] = d_final_flat

    save_dir = os.path.join(HERE, "poincare")
    os.makedirs(save_dir, exist_ok=True)
    
    out_path = os.path.join(save_dir, "poincare_sasrec_baseline_geometry_300x800_exp_c01.npz")
    np.savez(out_path, D=D_dense)
    print(f"[Save] Poincare geometry saved to {out_path}")

    # 3. Учим SASRec на ней (используем z_tangent)
    z_tangent_torch = torch.tensor(z_tangent_np, device=device)
    print("\n--- training SASRec on Poincare embeddings ---")
    hr_sasrec, ndcg_sasrec = train_and_eval_sasrec_on_fixed_Z(z_tangent_torch, data, device, args.latent_dim, sasrec_config, verbose=True)

    #print(f"\n--- [Hyper] training HypSASRec on Poincare embeddings ---",
    #          flush=True)
    #hr_h, ndcg_h = train_and_eval_hypsasrec_on_fixed_Z_nolog(
    #        z_tangent_torch, data, device, 64, sasrec_config,
    #        lr=1e-3, epochs=50, patience=8,
    #        verbose=True,
    #        c=50.0,
    #        freeze_items=True,
    #        freeze_scaler=True,
    #)
    
    print(f"\n=== POINCARE BASELINE SUMMARY ===")
    print(f"Downstream SASRec: HR@10={hr_sasrec:.4f}  NDCG@10={ndcg_sasrec:.4f}")
    #print(f"[Hyper] HR@10={hr_h:.4f}  NDCG@10={ndcg_h:.4f}", flush=True)

if __name__ == "__main__":
    main()
    