"""
Runs the eta_outer sweep on the ML-10M source at max_movies=1800 (chosen
after a density/timing analysis - see docs/recsys_paper_diary.md 2026-08-25:
at the paper's own max_movies=800, the 300 most-active ml-10m users are so
dense that 34/300 have fewer than 99 candidate negative items, degrading
evaluation; max_movies=1800 drops that to 2/300 while keeping full-sweep
wall-clock time to an estimated ~6.5h, extrapolated from a direct
IsomapNN forward+backward benchmark across candidate sizes).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

if __name__ == "__main__":
    sweep.main(
        n_run_prefix="ml10m_eta_sweep",
        max_users=300,
        max_movies=1800,
        dataset_dir_name="ml-10m",
        summary_filename="ml10m_eta_outer_sweep_summary.json",
    )
