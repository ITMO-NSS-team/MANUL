"""
ML-10M eta_outer sweep with patience=30, cf_epochs/final_cf_epochs cap=200.
See run_amazon_beauty_sweep_p30.py for why. Distinct n_run_prefix
(ml10m_eta_sweep_p30) so the hrfix results stay available for comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="ml10m_eta_sweep_p30",
    max_users=300,
    max_movies=1800,
    dataset_dir_name="ml-10m",
    dataset_type="movielens",
    select_by="hr",
    cf_epochs=200,
    final_cf_epochs=200,
    inner_patience=30,
    final_patience=30,
    summary_filename="ml10m_eta_outer_sweep_p30_summary.json",
)
