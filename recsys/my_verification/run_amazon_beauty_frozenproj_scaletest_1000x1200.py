"""
Scale-robustness test, safer scale (2026-09-13): after the 3000u/2500i
recon test hit a real instability (a gradient-guard trigger at outer step
20 was followed by a severe, silent slowdown - diagnosed via PowerShell
Get-Process as ~2 cores actively computing for 3+ hours with zero visible
progress, likely subnormal/denormalized-float arithmetic from
near-degenerate post-guard weights, not a deadlock), user asked to first
get one clean scale-robustness data point at a scale that never triggered
the guard in any prior probe: 1000u/1200i (~137s/step, confirmed clean).

Full protocol this time (not just recon): outer_epochs=300, matching the
validated 300u/800i multi-seed test, since the point is a genuine
scale-robustness data point, not just a quick trend check. Same config:
eta_outer=0.001, freeze_item_projection=True, dropout=0.2, select_by=hr,
inner_patience=30, final_patience=30, cap=200, seed=0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_frozenproj_scaletest_1000x1200",
    max_users=1000,
    max_movies=1200,
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
    seed=0,
    summary_filename="amazon_beauty_frozenproj_scaletest_1000x1200_summary.json",
)
