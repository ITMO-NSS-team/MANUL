"""
Validates the fixed-Z regularization finding (2026-08-30) in the actual
outer bilevel loop, not just the isolated fixed-Z diagnostic. dropout=0.2
(vs the previously-unexamined dropout=0.0 default) cut val_loss spread
across random seeds 8x (0.288 -> 0.037 relative) and test_hr std nearly in
half (0.038 -> 0.020) on a completely fixed manifold. Testing whether this
holds up when the manifold is also being optimized.

Same eta/horizon as the warm-start A/B test's control arm (eta=0.03,
outer_epochs=60, select_by=hr, patience=30, cap=200) - control arm's 5-seed
baseline (dropout=0.0) is already measured: test_hr=0.170+-0.018,
within-run val_hr std=0.0352+-0.0025. This run only changes dropout,
everything else identical, for direct comparison.

Usage: python run_amazon_beauty_dropout_test_seed.py --seed 0
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

n_run_prefix = f"amazon_beauty_dropout02_test_seed{args.seed}"
print(f"\n=== dropout=0.2 seed={args.seed} n_run_prefix={n_run_prefix} ===\n", flush=True)

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
    dropout=0.2,
    seed=args.seed,
    summary_filename=f"{n_run_prefix}_summary.json",
)
