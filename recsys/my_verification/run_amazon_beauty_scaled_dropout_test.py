"""
Step (2) of the "too many degrees of freedom" investigation (2026-08-30/
2026-09-09): step (1) confirmed dropout=0.2 substantially reduces
cross-seed variance (fixed-Z diagnostic: val_loss relative spread 0.288 ->
0.037; real outer loop, eta=0.03/60 steps: test_hr std 0.018 -> 0.009).
This step tests whether relaxing the data sparsity itself (the underlying
cause of underdetermination) further reduces the residual variance -
dropout regularizes a fixed, small, sparse problem; more data changes the
problem's actual identifiability.

Scale chosen empirically to fit a 6-8h budget on this GPU (RTX 5080,
16GB - memory is not the constraint at these sizes, wall-clock is):
- Probed 300u/800i (~120s/step, baseline), 1000u/1200i (~137s/step),
  3000u/2500i (~330s/step, clean gradients), 5000u/4000i (~459s/step, but
  the NaN/Inf gradient guard fired on EVERY step - 35843 non-finite
  entries zeroed each time, a structural eigenvalue-degeneracy issue at
  that item count, not a rare blip), 6000u/2500i (~455s/step, clean -
  comparable cost to 5000u/4000i but WITHOUT the instability, since items
  drive the eigenvalue-degeneracy risk while users don't).
- Chose users=6000, items=2500 (20x/3.1x the original 300/800): stays in
  the empirically-clean region for items while scaling users - and users
  are the more directly relevant axis for testing NCF's own
  underdetermination anyway (more interactions per user/item pair).
- outer_epochs=60 (same as all prior A/B tests in this investigation, for
  direct comparability) x ~455s/step + ~450s overhead ~= 27750s ~= 7.7h.

dropout=0.2 (the validated fix from step 1) kept on - this is the new
default going forward, not being re-tested here. select_by=hr,
patience=30, cap=200 unchanged. See docs/recsys_paper_diary.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_scaled6000x2500_dropout02",
    max_users=6000,
    max_movies=2500,
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
    seed=0,
    summary_filename="amazon_beauty_scaled6000x2500_dropout02_summary.json",
)
