"""
Reproduces the paper draft's core experiment (Section 4, Table 1): sweep
eta_outer in {0.01, 0.03, 0.05, 0.10} on the same 300-user/800-item
MovieLens-1M subsample, 30 outer epochs, computing the full geometry
diagnostics suite (geometry_diagnostics.py) at every outer epoch.

DEVIATION FROM THE PAPER'S EXACT TABLE 1 HYPERPARAMETERS (logged here
rather than silently matched): the paper specifies batch_size=32 and
inner_ep(cf_epochs)=100. At batch_size=32 over ~400k train pairs (positives
+ negatives), that's ~12.5k batches/inner-epoch x 100 inner epochs x 30
outer epochs x 4 configs - computationally infeasible in a reasonable
session (roughly 1000x the already-validated batch_size=2048 configuration
in wall-clock terms). Uses batch_size=2048 (already smoke-tested, see
docs/recsys_paper_diary.md 2026-08-24) with cf_epochs=30 as an early-stopping
CAP (GradientIsomapCF's inner loop already has early stopping, so this
governs worst case, not typical case). This is a real, acknowledged
deviation from exact hyperparameter fidelity, not an attempt to reproduce
Table 1's numbers bit-for-bit - see the paper draft update / diary for how
this affects interpretation of the comparison.
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment
from geometry_diagnostics import analyze_run

import pandas as pd


ETA_OUTER_VALUES = [0.01, 0.03, 0.05, 0.10]


def run_one(eta_outer: float, n_run_prefix: str = "eta_sweep"):
    n_run = f"{n_run_prefix}_{eta_outer}"
    print(f"\n{'=' * 70}")
    print(f"=== eta_outer = {eta_outer}  (n_run={n_run}) ===")
    print(f"{'=' * 70}", flush=True)

    t0 = time.time()
    results = run_experiment.main(
        max_users=300, max_movies=800, min_seq_len=2, num_ng=2, top_k=10,
        epochs_pure=0, run_ncf=False,   # baseline already established separately
        gradisomap_epochs=30, cf_epochs=30, final_cf_epochs=30,
        lr_isomap=eta_outer,
        run_gincf=True, n_run=n_run, seed=0,
    )
    elapsed = time.time() - t0
    print(f"[eta_outer={eta_outer}] finished in {elapsed:.1f}s", flush=True)

    logs_folder = os.path.join(HERE, "logs_movielens_isomap_cf", n_run)
    diag_results = analyze_run(logs_folder, max_epochs_to_show=None)
    diag_csv = os.path.join(logs_folder, "geometry_diagnostics.csv")
    pd.DataFrame(diag_results).to_csv(diag_csv, index=False)
    print(f"[Save] {diag_csv}", flush=True)

    return {
        "eta_outer": eta_outer,
        "n_run": n_run,
        "elapsed_s": elapsed,
        "test_hr": results["gincf"]["test_hr"],
        "test_ndcg": results["gincf"]["test_ndcg"],
        "had_bad_grad_any": any(results["gincf"]["outer_history"].get("had_bad_grad", [])),
        "logs_folder": logs_folder,
        "diag_csv": diag_csv,
    }


def main():
    summary = []
    for eta in ETA_OUTER_VALUES:
        summary.append(run_one(eta))
        # Save incrementally after each config, so a later config's failure
        # doesn't lose earlier results.
        with open(os.path.join(HERE, "eta_outer_sweep_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print("\n\n=== SWEEP SUMMARY ===")
    for row in summary:
        print(f"eta={row['eta_outer']:.2f}  HR@10={row['test_hr']:.4f}  "
              f"NDCG@10={row['test_ndcg']:.4f}  had_bad_grad={row['had_bad_grad_any']}  "
              f"time={row['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
