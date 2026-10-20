"""
Runs the eta_outer sweep on Amazon Reviews'23 Beauty_and_Personal_Care
(5-core) at the same scale as the paper's original ML-1M configuration
(300 users, 800 items) - the most direct "same pipeline, different domain"
comparison point, per the user's explicit choice of Amazon over Yelp
(2026-08-26, see docs/recsys_paper_diary.md).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_eta_sweep",
    max_users=300,
    max_movies=800,
    dataset_dir_name="amazon_beauty",
    dataset_type="amazon",
    amazon_category="Beauty_and_Personal_Care",
    summary_filename="amazon_beauty_eta_outer_sweep_summary.json",
)
