"""
Tests whether the outer loop's noisy trajectory is caused by too large an
AdamW step size on IsomapNN's weights (lr_isomap=eta_outer). Hypothesis
(user, 2026-08-29): each outer step perturbs the manifold by an amount
large enough to be undirected/unstructured, so the val_hr trajectory looks
like noise rather than a smooth trend. Tests two orders of magnitude below
the smallest eta tried so far (0.01): 0.001 and 0.0001.

outer_epochs=30 (cheap, matches the existing p30-generation 30-step runs
used for the eta=0.01/0.03/0.05/0.10 comparison, so this is directly
comparable) - inner loop settings unchanged (select_by=hr, patience=30,
cap=200, already validated not to hit the cap even at 200 outer steps).
See docs/recsys_paper_diary.md, 2026-08-29.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_smallerlr_test",
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
    outer_epochs=30,
    eta_values=[0.001, 0.0001],
    summary_filename="amazon_beauty_smallerlr_test_summary.json",
)
