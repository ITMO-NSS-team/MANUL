"""
Third seed repeat for the 3000u/2500i scale point (2026-09-16) - see
run_amazon_beauty_frozenproj_scaletest_3000x2500_seed1.py for context,
including the __main__-guard note (2026-09-17).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="amazon_beauty_frozenproj_scaletest_3000x2500_seed2",
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
        seed=2,
        diagnostics_workers=4,
        summary_filename="amazon_beauty_frozenproj_scaletest_3000x2500_seed2_summary.json",
    )
