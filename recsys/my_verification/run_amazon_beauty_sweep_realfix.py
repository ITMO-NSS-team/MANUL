"""
Reruns the Amazon Beauty eta_outer sweep with the REAL convergence fix -
select_by="hr" now genuinely drives both the inner-loop's stopping trigger
(not just checkpoint selection, see EarlyStopping.step()) and the outer
loop's final geometry (best-by-val_hr state is saved/restored, not just
whatever D_input the last outer step happened to land on). See
docs/recsys_paper_diary.md, 2026-08-27, and commit 391c525.

Quick test (single eta, cf_epochs=100 cap) showed the inner loop now
genuinely early-stops (mean ~29/100 epochs, range 20-52) instead of always
exhausting the cap, so this full sweep should run faster than the previous
"realfix"-less hrfix sweep despite the higher nominal cap.

Distinct n_run_prefix (amazon_beauty_realfix_eta_sweep) so the previous
hrfix results stay available for comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_realfix_eta_sweep",
    max_users=300,
    max_movies=800,
    dataset_dir_name="amazon_beauty",
    dataset_type="amazon",
    amazon_category="Beauty_and_Personal_Care",
    select_by="hr",
    cf_epochs=100,
    final_cf_epochs=100,
    inner_patience=8,
    final_patience=8,
    summary_filename="amazon_beauty_realfix_eta_outer_sweep_summary.json",
)
