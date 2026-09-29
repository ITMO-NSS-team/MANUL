"""
Same outer_epochs=200 confirmation test as run_amazon_beauty_outer200_test.py,
but for eta_outer=0.0001 - the smallest learning rate tested so far, which
at 30 outer steps already showed ~45% lower trajectory std than eta=0.01
(0.0225 vs 0.0415) and the best final test HR@10 (0.2100). Testing whether
a smaller outer step size reveals a real smooth trend over a longer horizon,
per the user's hypothesis (2026-08-29) that eta=0.01+ perturbs the manifold
too much per step to preserve directionality. See docs/recsys_paper_diary.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_outer200_test_eta0001",
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
    eta_values=[0.0001],
    summary_filename="amazon_beauty_outer200_test_eta0001_summary.json",
)
