"""
A/B test for the warm_start_inner option (2026-08-30): does continuing the
inner NCF proxy's training from the previous outer step's weights (instead
of a fresh random reinit every step) reduce the outer loop's step-to-step
noise? Motivated by a controlled experiment that held Z completely fixed
and found seed-only variance in test HR@10 (std=0.038) comparable to ALL
observed outer-loop noise (std 0.019-0.042 across every eta/horizon tested)
- pointing to per-step re-sampling of a random NCF local optimum ("basin")
as the likely dominant noise source, not the manifold's own movement.

Both arms use IDENTICAL settings (eta=0.03, outer_epochs=60, select_by=hr,
patience=30, cap=200) except warm_start_inner, run back-to-back so neither
is confounded by unrelated code changes. If the hypothesis is right,
`_warmstart` should show a visibly lower std(val_hr) / smaller mean
step-to-step train_loss swing than `_control`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

COMMON = dict(
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
)

print("\n\n########## ARM 1: control (fresh reinit every outer step) ##########\n", flush=True)
sweep.main(
    n_run_prefix="amazon_beauty_warmstart_ab_control",
    warm_start_inner=False,
    summary_filename="amazon_beauty_warmstart_ab_control_summary.json",
    **COMMON,
)

print("\n\n########## ARM 2: warm_start_inner=True ##########\n", flush=True)
sweep.main(
    n_run_prefix="amazon_beauty_warmstart_ab_warmstart",
    warm_start_inner=True,
    summary_filename="amazon_beauty_warmstart_ab_warmstart_summary.json",
    **COMMON,
)
