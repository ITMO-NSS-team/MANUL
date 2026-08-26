"""
Generalized version of full_hyperbolicity_table.py / _ml10m.py: builds the
same unified hyperbolicity+loss table for any eta_outer sweep run, given
its logs-folder prefix and a small downstream-metrics JSON. Introduced for
the third dataset (Amazon Beauty) rather than copy-pasting a third
near-duplicate script - see docs/recsys_paper_diary.md, 2026-08-26.

The downstream JSON has the shape:
{
  "pure_init": [val_loss, hr, ndcg],
  "<eta>": {"epoch0": [val_loss, hr, ndcg], "converged": [val_loss, hr, ndcg]},
  ...
  "poincare": [val_loss, hr, ndcg],
  "euclidean_baseline": [val_loss, hr, ndcg]
}
"""
import argparse
import glob
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta_prefix", required=True,
                        help="Folder prefix under logs_movielens_isomap_cf/, e.g. "
                             "'amazon_beauty_eta_sweep_' (folders are <prefix><eta>).")
    parser.add_argument("--etas", default="0.01,0.03,0.05,0.1")
    parser.add_argument("--downstream_json", required=True)
    parser.add_argument("--poincare_npz", required=True)
    parser.add_argument("--euclidean_npz", required=True)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGS = os.path.join(HERE, "logs_movielens_isomap_cf")
    etas = args.etas.split(",")

    with open(args.downstream_json) as f:
        downstream = json.load(f)

    rows = []

    # --- pure_init ---
    D_init = np.load(os.path.join(LOGS, f"{args.eta_prefix}{etas[0]}", "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(weights_initial_assumption=D_init_t, n_components=args.latent_dim,
                               n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure = pure_init_model().to(torch.float32).detach().cpu().numpy()
    D_latent_pure = np.linalg.norm(Z_pure[:, None, :] - Z_pure[None, :, :], axis=-1)
    vl, hr, ndcg = downstream["pure_init"]
    rows.append(diagnostics_for_D(D_latent_pure, "pure_init (euclidean, 0 outer steps)", vl, hr, ndcg))

    # --- GradientIsomapNCF epoch0 / epoch(last) per eta ---
    for eta in etas:
        cfg_folder = os.path.join(LOGS, f"{args.eta_prefix}{eta}")
        d0 = np.load(os.path.join(cfg_folder, "matrices_epoch0.npz"))["D_latent"]
        vl, hr, ndcg = downstream[eta]["epoch0"]
        rows.append(diagnostics_for_D(d0.astype(np.float64), f"GINCF eta={eta}, epoch0", vl, hr, ndcg))

        epoch_files = glob.glob(os.path.join(cfg_folder, "matrices_epoch*.npz"))
        last = max(epoch_files, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        d_last = np.load(last)["D_latent"]
        vl, hr, ndcg = downstream[eta]["converged"]
        rows.append(diagnostics_for_D(d_last.astype(np.float64), f"GINCF eta={eta}, converged", vl, hr, ndcg))

    # --- Poincare-pretrained and Euclidean-baseline geometries ---
    poincare_npz = np.load(os.path.join(HERE, args.poincare_npz))
    rows.append(diagnostics_for_D(
        poincare_npz["D"].astype(np.float64), "Poincare-pretrained (a priori hyperbolic)",
        float(poincare_npz["val_loss"]), float(poincare_npz["hr"]), float(poincare_npz["ndcg"])))

    euclidean_npz = np.load(os.path.join(HERE, args.euclidean_npz))
    rows.append(diagnostics_for_D(
        euclidean_npz["D"].astype(np.float64), "Euclidean NeuMF (geometry-free baseline)",
        float(euclidean_npz["val_loss"]), float(euclidean_npz["hr"]), float(euclidean_npz["ndcg"])))

    out_path = os.path.join(HERE, args.out_csv)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
