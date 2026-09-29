"""
Builds a single, consistent table of hyperbolicity measures + downstream
loss/metrics for every geometry compared in the paper's refined evaluation
(Section "A Refined Evaluation"): pure Euclidean-init Isomap Z, each
GradientIsomapNCF eta's epoch0/epoch29 Z, and the Poincare-pretrained
embedding. All measures are computed on D_latent = pairwise distances of
the actual Z fed to the downstream NCF head (not the D_input-based kNN
graph used elsewhere in geometry_diagnostics.py) - this is deliberately a
different, more targeted characterization: "what geometry does the
downstream model actually see", consistent across every row including
pure_init/Poincare which have no D_input-based kNN graph of their own.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MANUL_DIR = os.path.dirname(os.path.dirname(HERE))
GINCF_DIR = os.path.join(os.path.dirname(HERE), "GradIsomapCF_movielens")
sys.path.insert(0, MANUL_DIR)
sys.path.insert(0, GINCF_DIR)
sys.path.insert(0, HERE)

from Adam.Isomap import IsomapNN
from geometry_diagnostics import ollivier_ricci_curvature, graph_laplacian_spectral, persistent_homology
from analyze_hyperbolicity import delta_hyperbolicity


def knn_adj_from_dense(D, k=10):
    n = D.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        neighbors = np.argsort(D[i])[1:k + 1]
        for j in neighbors:
            adj[i, j] = D[i, j]
            adj[j, i] = D[j, i]
    return adj


def diagnostics_for_D(D, label, val_loss=None, hr=None, ndcg=None):
    n = D.shape[0]
    delta = delta_hyperbolicity(D.astype(np.float64), basepoint=0)
    diam = float(D.max())
    delta_rel = (2.0 * delta / diam) if diam > 0 else float("nan")

    knn_adj = knn_adj_from_dense(D, k=10)
    orc = ollivier_ricci_curvature(knn_adj)
    spec = graph_laplacian_spectral(knn_adj)
    ph = persistent_homology(D.astype(np.float64), max_edge_length=float(np.percentile(D, 3)))

    row = {
        "geometry": label,
        "delta": delta, "delta_rel": delta_rel,
        "orc_mean": orc["mean"], "orc_f_neg": orc["f_neg"],
        "lambda2": spec["lambda2"], "spectral_gap": spec["spectral_gap"],
        "h1_count": ph["h1_count"],
        "val_loss": val_loss, "hr10": hr, "ndcg10": ndcg,
    }
    print(f"{label:45s} delta_rel={delta_rel:.4f} ORC={orc['mean']:+.4f} f_neg={orc['f_neg']:.3f} "
          f"lambda2={spec['lambda2']:.4f} gap={spec['spectral_gap']:.4f} H1={ph['h1_count']:5d} "
          f"| val_loss={val_loss if val_loss is not None else float('nan'):.4f} "
          f"HR@10={hr if hr is not None else float('nan'):.4f} "
          f"NDCG@10={ndcg if ndcg is not None else float('nan'):.4f}", flush=True)
    return row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGS = os.path.join(HERE, "logs_movielens_isomap_cf")
    rows = []

    # Known downstream results from the corrected ablation + Poincare runs
    # (ablation_geometry_vs_optimization.py / poincare_baseline.py outputs,
    # 2026-08-25 - see docs/recsys_paper_diary.md).
    downstream = {
        "pure_init": (0.2962, 0.2767, 0.1565),
        "0.01_epoch0": (0.2946, 0.2867, 0.1684),
        "0.01_epoch29": (0.2908, 0.2567, 0.1519),
        "0.03_epoch0": (0.2990, 0.2667, 0.1665),
        "0.03_epoch29": (0.3222, 0.2400, 0.1227),
        "0.05_epoch0": (0.2986, 0.2533, 0.1565),
        "0.05_epoch29": (0.3289, 0.2433, 0.1325),
        "0.10_epoch0": (0.2990, 0.2700, 0.1652),
        "0.10_epoch29": (0.3356, 0.1833, 0.0949),
        "poincare": (0.3424, 0.2433, 0.1322),
        "euclidean_baseline": (0.2826, 0.3633, 0.2354),
    }

    # --- pure_init ---
    D_init = np.load(os.path.join(LOGS, "eta_sweep_0.01", "D_input_init.npy"))
    D_init_t = torch.tensor(D_init, dtype=torch.float32, device=device)
    pure_init_model = IsomapNN(weights_initial_assumption=D_init_t, n_components=64, n_neighbors=10).to(device)
    with torch.no_grad():
        Z_pure = pure_init_model().to(torch.float32).detach().cpu().numpy()
    D_latent_pure = np.linalg.norm(Z_pure[:, None, :] - Z_pure[None, :, :], axis=-1)
    vl, hr, ndcg = downstream["pure_init"]
    rows.append(diagnostics_for_D(D_latent_pure, "pure_init (euclidean, 0 outer steps)", vl, hr, ndcg))

    # --- GradientIsomapNCF epoch0 / epoch(last) per eta ---
    for eta in ["0.01", "0.03", "0.05", "0.1"]:
        cfg_folder = os.path.join(LOGS, f"eta_sweep_{eta}")
        d0 = np.load(os.path.join(cfg_folder, "matrices_epoch0.npz"))["D_latent"]
        eta_label = eta if len(eta) == 4 else eta + "0"
        vl, hr, ndcg = downstream.get(f"{eta_label}_epoch0", (None, None, None))
        rows.append(diagnostics_for_D(d0.astype(np.float64), f"GINCF eta={eta}, epoch0", vl, hr, ndcg))

        import glob, re
        epoch_files = glob.glob(os.path.join(cfg_folder, "matrices_epoch*.npz"))
        last = max(epoch_files, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        d_last = np.load(last)["D_latent"]
        vl, hr, ndcg = downstream.get(f"{eta_label}_epoch29", (None, None, None))
        rows.append(diagnostics_for_D(d_last.astype(np.float64), f"GINCF eta={eta}, converged", vl, hr, ndcg))

    # --- Poincare-pretrained and Euclidean-baseline geometries, computed
    # through the exact same diagnostics_for_D() pass as every other row
    # (previously these were only reported separately by poincare_baseline.py
    # / save_euclidean_baseline_geometry.py - folded in here for a single
    # unified table as requested). ---
    poincare_npz = np.load(os.path.join(HERE, "poincare_fitted_geometry.npz"))
    rows.append(diagnostics_for_D(
        poincare_npz["D"].astype(np.float64), "Poincare-pretrained (a priori hyperbolic)",
        float(poincare_npz["val_loss"]), float(poincare_npz["hr"]), float(poincare_npz["ndcg"])))

    euclidean_npz = np.load(os.path.join(HERE, "euclidean_baseline_geometry.npz"))
    rows.append(diagnostics_for_D(
        euclidean_npz["D"].astype(np.float64), "Euclidean NeuMF (geometry-free baseline)",
        float(euclidean_npz["val_loss"]), float(euclidean_npz["hr"]), float(euclidean_npz["ndcg"])))

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "full_hyperbolicity_table.csv"), index=False)
    print(f"\n[Save] {os.path.join(HERE, 'full_hyperbolicity_table.csv')}")


if __name__ == "__main__":
    main()
