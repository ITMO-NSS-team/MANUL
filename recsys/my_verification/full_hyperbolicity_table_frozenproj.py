"""
Hyperbolicity/geometry-diagnostics comparison table for the new winning
configuration (2026-09-11/12): eta_outer=0.001, freeze_item_projection=True,
dropout=0.2 - the first config in this whole investigation to show a
real, reproducible (3-seed) convergence trend in the outer loop's own
training loss, and to significantly beat a fair (dropout=0.2) Euclidean
baseline on test HR@10.

Differs from full_hyperbolicity_table_generic.py in one important way:
uses the RESTORED-BEST-BY-VAL_HR epoch's D_latent (the geometry the outer
loop actually kept and used to compute the reported test metrics), not
just the last saved epoch - now genuinely different (best epochs: 187,
216, 89 out of 300, never the last) since the outer loop's own restore-
best-not-last logic is central to this method.

Rows: pure_init (before any outer optimization), GINCF epoch0 vs
restored-best for each of the 3 seeds, Poincare-pretrained baseline, and
the FAIR (dropout=0.2) Euclidean baseline - not the older no-dropout one.
"""
import glob
import json
import os
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

LOGS = os.path.join(HERE, "logs_movielens_isomap_cf")
LATENT_DIM = 64

SEED_RUNS = {
    0: "amazon_beauty_frozenproj_eta001_outer300_seed0_0.001",
    1: "amazon_beauty_frozenproj_eta001_outer300_seed1_0.001",
    2: "amazon_beauty_frozenproj_eta001_outer300_seed2_0.001",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    # --- pure_init: D_input_init is seed-independent (built from the
    # fixed feature matrix, not from any RNG), use seed0's copy ---
    D_init = np.load(os.path.join(LOGS, SEED_RUNS[0], "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(weights_initial_assumption=D_init_t, n_components=LATENT_DIM,
                               n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure = pure_init_model().to(torch.float32).detach().cpu().numpy()
    D_latent_pure = np.linalg.norm(Z_pure[:, None, :] - Z_pure[None, :, :], axis=-1)
    rows.append(diagnostics_for_D(D_latent_pure, "pure_init (euclidean, 0 outer steps)"))

    # --- GINCF (frozen_item_projection + dropout=0.2 + eta=0.001), 3 seeds ---
    for seed, run in SEED_RUNS.items():
        cfg_folder = os.path.join(LOGS, run)

        d0 = np.load(os.path.join(cfg_folder, "matrices_epoch0.npz"))["D_latent"]
        rows.append(diagnostics_for_D(d0.astype(np.float64), f"GINCF seed={seed}, epoch0"))

        hist = np.load(os.path.join(cfg_folder, "history.npz"), allow_pickle=True)
        ep = hist["epoch"]
        val_hr = hist["val_hr"]
        val_loss = hist["val_loss"]
        best_idx = int(np.argmax(val_hr))
        best_epoch_num = int(ep[best_idx])

        with open(os.path.join(HERE, f"{run.replace('_0.001', '')}_summary.json")) as f:
            summ = json.load(f)
        test_hr = summ[0]["test_hr"]
        test_ndcg = summ[0]["test_ndcg"]

        best_file = os.path.join(cfg_folder, f"matrices_epoch{best_epoch_num}.npz")
        d_best = np.load(best_file)["D_latent"]
        rows.append(diagnostics_for_D(
            d_best.astype(np.float64), f"GINCF seed={seed}, restored-best (step {best_epoch_num+1}/300)",
            val_loss=float(val_loss[best_idx]), hr=test_hr, ndcg=test_ndcg))

    # --- Poincare-pretrained and the FAIR (dropout=0.2) Euclidean baseline ---
    poincare_npz = np.load(os.path.join(HERE, "poincare_fitted_geometry_amazon_beauty.npz"))
    rows.append(diagnostics_for_D(
        poincare_npz["D"].astype(np.float64), "Poincare-pretrained (a priori hyperbolic)",
        float(poincare_npz["val_loss"]), float(poincare_npz["hr"]), float(poincare_npz["ndcg"])))

    euclidean_npz = np.load(os.path.join(HERE, "euclidean_baseline_geometry_amazon_beauty_dropout02fair.npz"))
    rows.append(diagnostics_for_D(
        euclidean_npz["D"].astype(np.float64), "Euclidean NeuMF (dropout=0.2, fair baseline)",
        float(euclidean_npz["val_loss"]), float(euclidean_npz["hr"]), float(euclidean_npz["ndcg"])))

    out_path = os.path.join(HERE, "full_hyperbolicity_table_frozenproj.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
