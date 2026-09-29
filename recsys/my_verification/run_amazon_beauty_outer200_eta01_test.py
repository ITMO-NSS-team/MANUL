"""
Tests the opposite extreme of the outer-lr investigation (2026-08-30): the
user's correction to my earlier misread of "с еще большим шагом" - she meant
a LARGER outer lr (eta=0.1, "shake the manifold harder"), not more outer
steps. Rationale: since even the largest lr tried so far in the longer-
horizon tests (0.03) showed essentially zero step-by-step correlation
between geometry diagnostics and val_hr/val_loss (all |Spearman r|<0.2), a
much stronger perturbation might produce swings large enough to either (a)
reveal a real correlation if one exists, or (b) confirm more decisively
that geometry drift and downstream ranking quality are decoupled at this
dataset/model scale.

outer_epochs=200 (same horizon as the eta=0.03 and eta=0.0001 confirmation
tests, for direct comparability) - inner loop settings unchanged
(select_by=hr, patience=30, cap=200). See docs/recsys_paper_diary.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_outer200_eta01_test",
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
    eta_values=[0.1],
    summary_filename="amazon_beauty_outer200_eta01_test_summary.json",
)
