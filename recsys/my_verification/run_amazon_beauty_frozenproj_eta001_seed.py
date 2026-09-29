"""
Multi-seed repeat + extended horizon for the eta=0.001/freeze_item_projection
outer-loop test (2026-09-11), which showed the first statistically strong,
consistent-direction convergence trend in this whole investigation
(val_hr r^2=0.088 p=2e-5, train_loss r^2=0.20 p=2e-11, best checkpoint at
step 187/200 - late, not an early fluke). Single run only so far - user
asked to repeat properly given the whole investigation's standing lesson
that single runs aren't reliable evidence here.

outer_epochs raised 200->300: the first run's rolling-mean val_hr looked
like it may have been approaching a plateau near step 200 (last ~30 steps
flat around 0.18-0.19), not clearly still climbing - extending checks
whether that holds or whether it keeps improving with a longer horizon.

Same settings otherwise: eta=0.001, dropout=0.2, freeze_item_projection=True,
select_by=hr, patience=30, cap=200.

Usage: python run_amazon_beauty_frozenproj_eta001_seed.py --seed 1
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

n_run_prefix = f"amazon_beauty_frozenproj_eta001_outer300_seed{args.seed}"
print(f"\n=== eta=0.001 frozen seed={args.seed} n_run_prefix={n_run_prefix} ===\n", flush=True)

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
    outer_epochs=300,
    eta_values=[0.001],
    dropout=0.2,
    freeze_item_projection=True,
    seed=args.seed,
    summary_filename=f"{n_run_prefix}_summary.json",
)
