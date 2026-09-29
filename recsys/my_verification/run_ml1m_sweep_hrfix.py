"""
Repeats the original MovieLens-1M eta_outer sweep (300 users, 800 items -
the paper's primary configuration) with the corrected model-selection
criterion (select_by="hr", larger patience/epoch cap) for both the inner
per-outer-step NCF proxy and the final-NCF retraining stage - see
docs/recsys_paper_diary.md, 2026-08-26 night update, for why the original
loss-based criterion was a noisy model-selection signal and why the
Amazon Beauty resweep with this fix changed which eta wins there.

Uses a distinct n_run_prefix (eta_sweep_hrfix) so this does not overwrite
the original eta_sweep_* results until reviewed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="eta_sweep_hrfix",
    max_users=300,
    max_movies=800,
    dataset_dir_name="ml-1m",
    dataset_type="movielens",
    select_by="hr",
    cf_epochs=60,
    final_cf_epochs=60,
    inner_patience=8,
    final_patience=8,
    summary_filename="eta_outer_sweep_hrfix_summary.json",
)
