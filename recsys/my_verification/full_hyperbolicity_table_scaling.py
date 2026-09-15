"""
Extends full_hyperbolicity_table_frozenproj.py's comparison (pure_init,
GINCF 3-seed @ 300u/800i, Poincare-pretrained, fair Euclidean baseline)
with the two larger-scale GINCF scale-test runs (1000u/1200i, 3000u/2500i)
under the exact same adopted config (freeze_item_projection=True,
dropout=0.2, eta_outer=0.001), using the same restored-best-by-val_hr
epoch selection and the same diagnostics_for_D() pass for an apples-to-
apples comparison across scales. See docs/recsys_paper_diary.md, 2026-09-15.

Each larger-scale point is currently n=1 (no seed repeats yet) - plotted
and labeled as such, not averaged/error-barred.
"""
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

from full_hyperbolicity_table import diagnostics_for_D

LOGS = os.path.join(HERE, "logs_movielens_isomap_cf")

SCALE_RUNS = {
    "1000u/1200i": "amazon_beauty_frozenproj_scaletest_1000x1200_0.001",
    "3000u/2500i": "amazon_beauty_frozenproj_scaletest_3000x2500_0.001",
}


def restored_best_row(scale_label, run):
    cfg_folder = os.path.join(LOGS, run)

    hist = np.load(os.path.join(cfg_folder, "history.npz"), allow_pickle=True)
    ep = hist["epoch"]
    val_hr = hist["val_hr"]
    val_loss = hist["val_loss"]
    best_idx = int(np.argmax(val_hr))
    best_epoch_num = int(ep[best_idx])

    with open(os.path.join(HERE, f"results_{run}.json")) as f:
        summ = json.load(f)
    test_hr = summ["gincf"]["test_hr"]
    test_ndcg = summ["gincf"]["test_ndcg"]

    n_total = len(ep)
    best_file = os.path.join(cfg_folder, f"matrices_epoch{best_epoch_num}.npz")
    d_best = np.load(best_file)["D_latent"]
    return diagnostics_for_D(
        d_best.astype(np.float64),
        f"GINCF {scale_label}, restored-best (step {best_epoch_num + 1}/{n_total})",
        val_loss=float(val_loss[best_idx]), hr=test_hr, ndcg=test_ndcg,
    )


def main(only=None):
    """only: optional list of scale labels to (re)compute - lets the two
    scale points be run separately (e.g. to avoid CPU contention with the
    concurrently-running geometry_diagnostics.py analyze_run job at the
    same n=2500 scale). Existing rows for labels not in `only` are kept."""
    out_path = os.path.join(HERE, "full_hyperbolicity_table_scaling.csv")
    existing = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    existing_by_scale = {r["geometry"].split(",")[0].replace("GINCF ", ""): r for r in existing}

    rows = []
    for scale_label, run in SCALE_RUNS.items():
        if only is not None and scale_label not in only:
            if scale_label in existing_by_scale:
                rows.append(existing_by_scale[scale_label])
            continue
        print(f"\n=== {scale_label} ===", flush=True)
        rows.append(restored_best_row(scale_label, run))
        pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    main(only=only)
