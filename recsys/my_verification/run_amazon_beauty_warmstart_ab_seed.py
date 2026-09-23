"""
Parameterized single (arm, seed) run for the warm-start A/B multi-repeat
test (2026-08-30). Takes --arm {control,warmstart} and --seed on the CLI so
multiple independent OS processes can be launched in parallel (one per
arm/seed combination) instead of running everything sequentially in one
process.

Same settings as run_amazon_beauty_warmstart_ab_test.py's single-run A/B
test (eta=0.03, outer_epochs=60, select_by=hr, patience=30, cap=200) -
only warm_start_inner and seed vary. Distinct n_run_prefix per (arm, seed)
so all repeats' logs/snapshots persist independently.

Usage: python run_amazon_beauty_warmstart_ab_seed.py --arm control --seed 1
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=["control", "warmstart"])
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

n_run_prefix = f"amazon_beauty_warmstart_ab_{args.arm}_seed{args.seed}"
print(f"\n=== arm={args.arm} seed={args.seed} n_run_prefix={n_run_prefix} ===\n", flush=True)

sweep.main(
    n_run_prefix=n_run_prefix,
    max_users=300,
    max_movies=800,
    dataset_dir_name="amazon_beauty",
    dataset_type="amazon",
    amazon_category="Beauty_and_Personal_Care",
    select_by="hr",
    cf_epochs=200,
    final_cf_epochs=200,
    inner_patience=30,
    final_patience=30,
    outer_epochs=60,
    eta_values=[0.03],
    warm_start_inner=(args.arm == "warmstart"),
    seed=args.seed,
    summary_filename=f"{n_run_prefix}_summary.json",
)
