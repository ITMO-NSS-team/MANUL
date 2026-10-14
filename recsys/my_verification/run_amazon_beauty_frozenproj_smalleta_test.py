"""
Revisits the "does a smaller outer lr smooth the trajectory into a real
trend" question (previously tested exhaustively under the UNFROZEN
architecture, down to eta=0.00001/500 steps, never significant: r^2=0.001,
p=0.48) - but now under freeze_item_projection=True, where the inner
critic has been confirmed to actually depend on the manifold's geometry
(real Z significantly beats shuffled Z, p=0.0005 on the fixed-Z
diagnostic). User's hypothesis: a smaller outer step should perturb the
manifold less per step, so the geometry fed into NCF varies less between
consecutive outer steps, and the (now geometry-sensitive) critic's
gradient signal should show a more pronounced direction, even if slower.

eta=0.03/60 steps under freeze_item_projection already tested (r^2=0.034,
p=0.158, not significant, comparable to or weaker than the unfrozen
control at the same horizon). This test: eta=0.001 (10x smaller),
outer_epochs=200 (matches the standard horizon used for the eta=0.1/
eta=0.0001 unfrozen probes, for direct comparability), same other
settings (dropout=0.2, select_by=hr, patience=30, cap=200).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_frozenproj_eta001_outer200",
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
    eta_values=[0.001],
    dropout=0.2,
    freeze_item_projection=True,
    seed=0,
    summary_filename="amazon_beauty_frozenproj_eta001_outer200_summary.json",
)
