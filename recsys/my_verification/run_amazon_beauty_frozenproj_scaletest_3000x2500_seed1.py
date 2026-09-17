"""
Second seed repeat for the 3000u/2500i scale point (2026-09-16), matching
run_amazon_beauty_frozenproj_scaletest.py's config exactly (seed=0 run,
outer_epochs=100) except seed=1 - requested so the scale-comparison plot
has the same 3-seed robustness picture at this scale that 300u/800i
already has. See docs/recsys_paper_diary.md, 2026-09-16.

NOTE (2026-09-17): guarded with __main__ - see
run_amazon_beauty_frozenproj_scaletest_1000x1200_seed1.py's docstring for
why this matters (fork-and-crash loop when diagnostics_workers>1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="amazon_beauty_frozenproj_scaletest_3000x2500_seed1",
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
        seed=1,
        diagnostics_workers=4,
        summary_filename="amazon_beauty_frozenproj_scaletest_3000x2500_seed1_summary.json",
    )
