"""
Amazon Beauty eta_outer sweep with patience=30 (up from 8), cf_epochs/
final_cf_epochs cap=200 (up from 100) - both inner-loop AND final-NCF
patience raised together per the user's request, after a single-eta test
(eta=0.03) showed patience=8 was cutting training off before a clean
overfitting peak was reached (val HR@10 peaked at inner epoch 34, patience=8
would stop at ~42 with the same peak already captured, but patience=30
lets the full rise-plateau-overfit cycle be seen and confirms the
restored checkpoint is genuine). See docs/recsys_paper_diary.md,
2026-08-27 night.

Distinct n_run_prefix (amazon_beauty_p30_eta_sweep) so the realfix results
stay available for comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_p30_eta_sweep",
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
    summary_filename="amazon_beauty_p30_eta_outer_sweep_summary.json",
)
