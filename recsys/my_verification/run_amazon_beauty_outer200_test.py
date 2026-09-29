"""
Single-eta confirmation test: outer loop run for 200 steps instead of 30
(inner loop keeps the already-validated select_by="hr", patience=30, cap=200),
to check whether the outer trajectory shows a real trend over a much longer
horizon, or stays noisy as it did over 30 steps - and whether the inner-loop
NCF proxy converges well at every one of those 200 outer steps regardless of
which manifold D_input it's being asked to fit. Requested by the user after
reviewing the p30-generation outer-loop plots. See docs/recsys_paper_diary.md.

Single eta (0.03, the one used for all prior single-eta tests) to keep this
cheap before committing to a 4-eta sweep at outer_epochs=200.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_outer200_test",
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
    outer_epochs=200,
    eta_values=[0.03],
    summary_filename="amazon_beauty_outer200_test_summary.json",
)
