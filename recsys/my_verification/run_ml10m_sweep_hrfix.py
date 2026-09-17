"""
Repeats the ML-10M eta_outer sweep (300 users, 1800 items) with the
corrected model-selection criterion (select_by="hr", larger patience/epoch
cap) - see docs/recsys_paper_diary.md, 2026-08-26 night update.

Uses a distinct n_run_prefix (ml10m_eta_sweep_hrfix) so this does not
overwrite the original ml10m_eta_sweep_* results until reviewed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="ml10m_eta_sweep_hrfix",
        max_users=300,
        max_movies=1800,
        dataset_dir_name="ml-10m",
        dataset_type="movielens",
        select_by="hr",
        cf_epochs=60,
        final_cf_epochs=60,
        inner_patience=8,
        final_patience=8,
        summary_filename="ml10m_eta_outer_sweep_hrfix_summary.json",
    )
