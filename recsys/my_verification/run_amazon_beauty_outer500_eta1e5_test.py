"""
Confirmation test after the outer-lr investigation (2026-08-29): the user
rejected warm-starting the inner NCF proxy (a new manifold every outer step
means old weights would only hurt, not help) and instead asked for (a) an
even smaller outer lr, (b) more than 200 outer steps, (c) a DataLoader/cudnn
determinism fix so repeated runs of the same config are actually comparable
(see GradientIsomapCF_log.py's _build_dataloader_and_full_tensors - the
inter_loader's shuffle previously used the unseeded global RNG stream, now
uses an explicit generator seeded from ng_seed; run_experiment.main also now
sets cudnn.deterministic=True and use_deterministic_algorithms(warn_only=True)).

eta_outer=0.00001 (10x below the smallest tried so far, 0.0001), 500 outer
steps (well above 200) - same inner settings (select_by=hr, patience=30,
cap=200, already validated not to hit the cap at up to 200 outer steps).
Expensive: ~8h at the ~59s/outer-step rate observed for eta=0.0001/outer=200.
See docs/recsys_paper_diary.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eta_outer_sweep as sweep

sweep.main(
    n_run_prefix="amazon_beauty_outer500_eta1e5_test",
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
    outer_epochs=500,
    eta_values=[0.00001],
    summary_filename="amazon_beauty_outer500_eta1e5_test_summary.json",
)
