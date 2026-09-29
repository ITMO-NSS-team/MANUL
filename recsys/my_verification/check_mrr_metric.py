"""
Sanity checks for evaluation.py's mrr_metric and its wiring into
evaluate_topk_pure/evaluate_topk_isomap (return_mrr=False by default, so
existing callers - main_new2.py, run_experiment.py,
run_eta_outer_sweep.py - keep getting a 2-tuple unchanged).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GradIsomapCF_movielens"))
from evaluation import mrr_metric, ndcg_metric, hit_metric


def check_mrr_arithmetic():
    # ground truth at rank 1 (idx=0): MRR=1.0, same as NDCG (log2(2)=1)
    assert mrr_metric(5, [5, 1, 2]) == 1.0
    assert ndcg_metric(5, [5, 1, 2]) == 1.0

    # ground truth at rank 2 (idx=1): MRR=1/2=0.5, NDCG=1/log2(3)=0.6309...
    # - this is the case that shows MRR and NDCG are genuinely different
    # metrics, not just two names for the same number.
    assert mrr_metric(5, [1, 5, 2]) == 0.5
    ndcg_rank2 = ndcg_metric(5, [1, 5, 2])
    assert abs(ndcg_rank2 - 0.6309297535714575) < 1e-9
    assert mrr_metric(5, [1, 5, 2]) != ndcg_metric(5, [1, 5, 2]), \
        "MRR and NDCG must diverge for a hit below rank 1 - confirms they are distinct metrics"

    # ground truth absent: both 0, same as HR
    assert mrr_metric(5, [1, 2, 3]) == 0.0
    assert hit_metric(5, [1, 2, 3]) == 0

    print("check_mrr_arithmetic: PASSED")


def check_evaluate_functions_backward_compatible():
    # Import-time check only (no GPU/model needed): confirm the new
    # return_mrr parameter exists and defaults to False without breaking
    # the function signatures existing callers rely on.
    import inspect
    from evaluation import evaluate_topk_pure, evaluate_topk_isomap

    sig_pure = inspect.signature(evaluate_topk_pure)
    sig_iso = inspect.signature(evaluate_topk_isomap)
    assert sig_pure.parameters["return_mrr"].default is False
    assert sig_iso.parameters["return_mrr"].default is False
    print("check_evaluate_functions_backward_compatible: PASSED")


if __name__ == "__main__":
    check_mrr_arithmetic()
    check_evaluate_functions_backward_compatible()
    print("All evaluation-metric sanity checks PASSED.")
