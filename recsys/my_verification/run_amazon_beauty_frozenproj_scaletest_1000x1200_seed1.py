"""
Second seed repeat for the 1000u/1200i scale point (2026-09-16), matching
run_amazon_beauty_frozenproj_scaletest_1000x1200.py's config exactly
(seed=0 run) except seed=1 - requested so the scale-comparison plot has
the same 3-seed robustness picture at this scale that 300u/800i already
has. See docs/recsys_paper_diary.md, 2026-09-16.

NOTE (2026-09-17): this and every other run_*.py driver script in this
file must guard its top-level sweep.main(...) call with
`if __name__ == "__main__":`. Without it, when diagnostics_workers>1
causes geometry_diagnostics.py's analyze_run to spawn a
multiprocessing.Pool, Windows' spawn start method re-imports this script
in each worker - and an unguarded top-level call re-runs the ENTIRE
bilevel training pipeline in every "worker". Each such worker quickly
dies (GPU/CPU contention -> MemoryError), multiprocessing.Pool silently
replaces dead workers, and the whole thing repeats forever: a self-
sustaining fork-and-crash loop that burned ~20h of GPU time across 3
runs before being caught (see docs/recsys_paper_diary.md, 2026-09-17).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="amazon_beauty_frozenproj_scaletest_1000x1200_seed1",
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
        seed=1,
        diagnostics_workers=4,
        summary_filename="amazon_beauty_frozenproj_scaletest_1000x1200_seed1_summary.json",
    )
