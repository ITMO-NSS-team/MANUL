"""
Extends the outer_epochs=500/eta=0.00001 run to 1000 outer steps, same
config. Motivation (user, 2026-08-30): after confirming that geometry
diagnostics DO show a strong, statistically robust directed trend over 500
outer steps (r^2 up to 0.89) while val_hr shows none (r^2=0.001, p=0.48),
and that no geometry metric correlates step-by-step with val_hr/val_loss
either (Spearman |r|<0.2 for all 11 metrics across 3 runs), the next
question is whether val_hr's trend is simply too weak to detect at n=500
and needs a longer horizon to resolve from noise - the same way the
geometry trend only became clearly significant at longer horizons.

Same settings as run_amazon_beauty_outer500_eta1e5_test.py (select_by=hr,
patience=30, cap=200) - only outer_epochs raised 500->1000. Very expensive:
~18h at the ~66s/outer-step rate observed for the 500-step run (32800s/500).
See docs/recsys_paper_diary.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_outer1000_eta1e5_test",
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
    outer_epochs=1000,
    eta_values=[0.00001],
    summary_filename="amazon_beauty_outer1000_eta1e5_test_summary.json",
)
