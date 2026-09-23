"""
Same unified hyperbolicity+loss table as full_hyperbolicity_table.py, but
for the second-scale (MovieLens-10M, 300 users / 1800 items) validation in
main.tex's subsec:ml10m. Reuses diagnostics_for_D()/knn_adj_from_dense()
from full_hyperbolicity_table.py so both tables are computed identically;
only the folder layout and the (corrected, checkpoint-bug-fixed) downstream
numbers differ - see docs/recsys_paper_diary.md, 2026-08-25 evening update.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MANUL_DIR = os.path.dirname(os.path.dirname(HERE))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, MANUL_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from Adam.Isomap import IsomapNN
from full_hyperbolicity_table import diagnostics_for_D


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGS = os.path.join(HERE, "logs_movielens_isomap_cf")
    rows = []

    # Corrected (checkpoint-bug-fixed) downstream numbers from
    # ablation_geometry_vs_optimization.py --dataset_dir_name ml-10m
    # --max_movies 1800, 2026-08-25 evening - see the diary. Format:
    # (val_loss, hr, ndcg).
    downstream = {
        "pure_init": (0.2904, 0.2542, 0.1394),
    }
    # Per-eta (val_loss, hr, ndcg) at epoch0 / converged, keyed by eta string.
    per_eta = {
        "0.01": {"epoch0": (0.2904, 0.2341, 0.1296), "converged": (0.2719, 0.2910, 0.1658)},
        "0.03": {"epoch0": (0.2903, 0.2676, 0.1429), "converged": (0.3005, 0.2475, 0.1261)},
        "0.05": {"epoch0": (0.2854, 0.2910, 0.1771), "converged": (0.3144, 0.2408, 0.1265)},
        "0.1":  {"epoch0": (0.2891, 0.2375, 0.1362), "converged": (0.3182, 0.2107, 0.1047)},
    }

    # --- pure_init ---
    D_init = np.load(os.path.join(LOGS, "ml10m_eta_sweep_0.01", "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(weights_initial_assumption=D_init_t, n_components=64, n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure = pure_init_model().to(torch.float32).detach().cpu().numpy()
    D_latent_pure = np.linalg.norm(Z_pure[:, None, :] - Z_pure[None, :, :], axis=-1)
    vl, hr, ndcg = downstream["pure_init"]
    rows.append(diagnostics_for_D(D_latent_pure, "pure_init (euclidean, 0 outer steps)", vl, hr, ndcg))

    # --- GradientIsomapNCF epoch0 / epoch(last) per eta ---
    for eta in ["0.01", "0.03", "0.05", "0.1"]:
        cfg_folder = os.path.join(LOGS, f"ml10m_eta_sweep_{eta}")
        d0 = np.load(os.path.join(cfg_folder, "matrices_epoch0.npz"))["D_latent"]
        vl, hr, ndcg = per_eta[eta]["epoch0"]
        rows.append(diagnostics_for_D(d0.astype(np.float64), f"GINCF eta={eta}, epoch0", vl, hr, ndcg))

        epoch_files = glob.glob(os.path.join(cfg_folder, "matrices_epoch*.npz"))
        last = max(epoch_files, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        d_last = np.load(last)["D_latent"]
        vl, hr, ndcg = per_eta[eta]["converged"]
        rows.append(diagnostics_for_D(d_last.astype(np.float64), f"GINCF eta={eta}, converged", vl, hr, ndcg))

    # --- Poincare-pretrained and Euclidean-baseline geometries ---
    poincare_npz = np.load(os.path.join(HERE, "poincare_fitted_geometry_ml10m.npz"))
    rows.append(diagnostics_for_D(
        poincare_npz["D"].astype(np.float64), "Poincare-pretrained (a priori hyperbolic)",
        float(poincare_npz["val_loss"]), float(poincare_npz["hr"]), float(poincare_npz["ndcg"])))

    euclidean_npz = np.load(os.path.join(HERE, "euclidean_baseline_geometry_ml10m.npz"))
    rows.append(diagnostics_for_D(
        euclidean_npz["D"].astype(np.float64), "Euclidean NeuMF (geometry-free baseline)",
        float(euclidean_npz["val_loss"]), float(euclidean_npz["hr"]), float(euclidean_npz["ndcg"])))

    out_path = os.path.join(HERE, "full_hyperbolicity_table_ml10m.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
