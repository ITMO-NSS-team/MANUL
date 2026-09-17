"""
Extends full_hyperbolicity_table_frozenproj.py's comparison (pure_init,
GINCF 3-seed @ 300u/800i, Poincare-pretrained, fair Euclidean baseline)
with the two larger-scale GINCF scale-test runs (1000u/1200i, 3000u/2500i)
under the exact same adopted config (freeze_item_projection=True,
dropout=0.2, eta_outer=0.001), using the same restored-best-by-val_hr
epoch selection and the same diagnostics_for_D() pass for an apples-to-
apples comparison across scales. See docs/recsys_paper_diary.md, 2026-09-15
and 2026-09-16/17 (seed repeats added for "equal conditions" across scales).
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

# scale_label -> {seed: n_run folder name}
SCALE_SEED_RUNS = {
    "1000u/1200i": {
        0: "amazon_beauty_frozenproj_scaletest_1000x1200_0.001",
        1: "amazon_beauty_frozenproj_scaletest_1000x1200_seed1_0.001",
        2: "amazon_beauty_frozenproj_scaletest_1000x1200_seed2_0.001",
    },
    "3000u/2500i": {
        0: "amazon_beauty_frozenproj_scaletest_3000x2500_0.001",
        1: "amazon_beauty_frozenproj_scaletest_3000x2500_seed1_0.001",
        2: "amazon_beauty_frozenproj_scaletest_3000x2500_seed2_0.001",
    },
}


def restored_best_row(scale_label, seed, run):
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
        f"GINCF {scale_label} seed={seed}, restored-best (step {best_epoch_num + 1}/{n_total})",
        val_loss=float(val_loss[best_idx]), hr=test_hr, ndcg=test_ndcg,
    )


# scale_label -> npz tag suffix used by poincare_baseline.py / save_euclidean_baseline_geometry.py
SCALE_BASELINE_TAGS = {
    "1000u/1200i": "amazon_beauty_1000x1200",
    "3000u/2500i": "amazon_beauty_3000x2500",
}


def poincare_row(scale_label, tag):
    npz = np.load(os.path.join(HERE, f"poincare_fitted_geometry_{tag}.npz"))
    return diagnostics_for_D(
        npz["D"].astype(np.float64), f"Poincare-pretrained {scale_label}",
        float(npz["val_loss"]), float(npz["hr"]), float(npz["ndcg"]))


def euclidean_row(scale_label, tag):
    npz = np.load(os.path.join(HERE, f"euclidean_baseline_geometry_{tag}_dropout02fair.npz"))
    return diagnostics_for_D(
        npz["D"].astype(np.float64), f"Euclidean NeuMF (dropout=0.2, fair) {scale_label}",
        float(npz["val_loss"]), float(npz["hr"]), float(npz["ndcg"]))


def _row_key(geometry_label):
    # "GINCF 1000u/1200i seed=0, restored-best (...)" -> "1000u/1200i seed=0"
    # "Poincare-pretrained 1000u/1200i" -> "Poincare-pretrained 1000u/1200i" (unchanged)
    return geometry_label.split(", restored-best")[0].replace("GINCF ", "")


def main(only_scales=None, only_seeds=None, include_baselines=True):
    """only_scales/only_seeds: optional filters so a subset can be (re)computed
    without redoing already-saved rows (e.g. to avoid CPU contention with a
    concurrently-running training job, or to add a seed as soon as it lands
    without recomputing the others)."""
    out_path = os.path.join(HERE, "full_hyperbolicity_table_scaling.csv")
    existing = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    existing_by_key = {_row_key(r["geometry"]): r for r in existing}

    rows_by_key = dict(existing_by_key)
    for scale_label, seed_runs in SCALE_SEED_RUNS.items():
        if only_scales is not None and scale_label not in only_scales:
            continue
        for seed, run in seed_runs.items():
            if only_seeds is not None and seed not in only_seeds:
                continue
            key = f"{scale_label} seed={seed}"
            cfg_folder = os.path.join(LOGS, run)
            if not os.path.exists(os.path.join(cfg_folder, "history.npz")):
                print(f"[skip] {key}: no history.npz yet (training not finished)", flush=True)
                continue
            print(f"\n=== {key} ===", flush=True)
            rows_by_key[key] = restored_best_row(scale_label, seed, run)
            pd.DataFrame(list(rows_by_key.values())).to_csv(out_path, index=False)

        if include_baselines and (only_scales is None or scale_label in only_scales):
            tag = SCALE_BASELINE_TAGS[scale_label]
            print(f"\n=== Poincare-pretrained {scale_label} ===", flush=True)
            rows_by_key[f"Poincare-pretrained {scale_label}"] = poincare_row(scale_label, tag)
            print(f"\n=== Euclidean baseline {scale_label} ===", flush=True)
            rows_by_key[f"Euclidean NeuMF (dropout=0.2, fair) {scale_label}"] = euclidean_row(scale_label, tag)
            pd.DataFrame(list(rows_by_key.values())).to_csv(out_path, index=False)

    pd.DataFrame(list(rows_by_key.values())).to_csv(out_path, index=False)
    print(f"\n[Save] {out_path}")


if __name__ == "__main__":
    main()
