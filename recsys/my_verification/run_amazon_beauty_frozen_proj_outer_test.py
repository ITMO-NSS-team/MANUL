"""
The decisive next step (2026-09-11): does the outer bilevel loop start to
show a real convergence trend once freeze_item_projection=True closes the
"escape hatch" that let NCF ignore the manifold's actual geometry? All
prior outer-loop tests (eta 0.00001-0.1, 30-500 steps, with and without
dropout=0.2) showed val_hr trend r^2<=0.04, never statistically
significant - but those all used the UNFROZEN NeuMFOnManifold, where the
fixed-Z diagnostic later showed real vs shuffled-Z made no significant
difference either (the network could ignore geometry entirely). With
freeze_item_projection=True, the same diagnostic showed real Z
significantly beats shuffled Z (p=0.0005) - so NOW, for the first time,
the outer loop's gradient signal is computed through a critic that
actually depends on Z's geometry, not just an arbitrary reinterpretation
of it. If the outer loop still doesn't converge under this condition, the
noise is coming from somewhere else (the fresh-reinit-per-step multimodal
NCF training itself, already documented) rather than "gradient computed
through a critic blind to geometry."

Same eta=0.03/outer_epochs=60 protocol as every other A/B test in this
investigation, for direct comparability. dropout=0.2 (validated),
freeze_item_projection=True (new). select_by=hr, patience=30, cap=200.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_frozenproj_outer_test",
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
    outer_epochs=60,
    eta_values=[0.03],
    dropout=0.2,
    freeze_item_projection=True,
    seed=0,
    summary_filename="amazon_beauty_frozenproj_outer_test_summary.json",
)
