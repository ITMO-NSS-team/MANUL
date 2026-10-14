"""
Repeats the Amazon Beauty eta_outer sweep with the corrected model-selection
criterion (select_by="hr", larger patience/epoch cap) for both the inner
per-outer-step NCF proxy and the final-NCF retraining stage - see
docs/recsys_paper_diary.md, 2026-08-26 evening update, for why the original
loss-based criterion was a noisy model-selection signal (train/val
negative-sampling ratio mismatch) and why fixing it was verified to widen,
not close, GradientIsomapNCF's lead over the Euclidean baseline.

Uses a distinct n_run_prefix (amazon_beauty_hrfix_eta_sweep) so this does not
overwrite the original amazon_beauty_eta_sweep_* results until reviewed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_hrfix_eta_sweep",
    max_users=300,
    max_movies=800,
    dataset_dir_name="amazon_beauty",
    dataset_type="amazon",
    amazon_category="Beauty_and_Personal_Care",
    select_by="hr",
    cf_epochs=60,
    final_cf_epochs=60,
    inner_patience=8,
    final_patience=8,
    summary_filename="amazon_beauty_hrfix_eta_outer_sweep_summary.json",
)
