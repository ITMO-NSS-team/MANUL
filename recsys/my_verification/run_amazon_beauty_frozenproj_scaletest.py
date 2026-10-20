"""
Scale-robustness reconnaissance for the adopted method (2026-09-12):
freeze_item_projection=True + dropout=0.2 + eta_outer=0.001, confirmed
at 300u/800i (3-seed: test_hr=0.194+-0.009, beats fair baseline
p=0.0075, train_loss trend significant in all 3 seeds). This exact
config's behavior at a larger data scale is UNVERIFIED - the one prior
large-scale run (6000u/2500i, HR@10=0.2827) used only dropout=0.2, no
freeze_item_projection, single seed, predates this whole line of fixes.

Scale chosen from prior timing/stability probes: 3000u/2500i (~330s/step,
confirmed clean - no NaN/Inf gradient guard triggers, unlike 5000u/4000i
which triggered the guard on every step). eta=0.001 needs ~200-300 outer
steps to show its convergence trend (established at 300u/800i - 60 steps
wasn't enough even under freeze_item_projection). At this larger scale,
300 steps would cost ~27-28h - too expensive to commit to blind.

This is a reconnaissance run: only 100 outer steps (~9h), single seed,
to see whether a trend starts to emerge before committing to the full
horizon. Same other settings: select_by=hr, inner_patience=30,
final_patience=30, cap=200.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_frozenproj_scaletest_3000x2500",
    max_users=3000,
    max_movies=2500,
    dataset_dir_name="amazon_beauty",
    dataset_type="amazon",
    amazon_category="Beauty_and_Personal_Care",
    select_by="hr",
    cf_epochs=200,
    final_cf_epochs=200,
    inner_patience=30,
    final_patience=30,
    outer_epochs=100,
    eta_values=[0.001],
    dropout=0.2,
    freeze_item_projection=True,
    seed=0,
    summary_filename="amazon_beauty_frozenproj_scaletest_3000x2500_summary.json",
)
