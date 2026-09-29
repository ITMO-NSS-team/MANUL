import torch

import numpy as np


def hit_metric(gt_item, pred_items):
    return 1 if gt_item in pred_items else 0


def ndcg_metric(gt_item, pred_items):
    if gt_item in pred_items:
        idx = pred_items.index(gt_item)
        return 1.0 / np.log2(idx + 2)
    return 0.0


def mrr_metric(gt_item, pred_items):
    """Reciprocal rank of the ground-truth item (0 if absent from pred_items).

    Distinct from ndcg_metric: NDCG here is log2-discounted (1/log2(idx+2)),
    MRR is linear (1/(idx+1)) - MRR penalizes a hit at rank 2 vs rank 1 more
    steeply than NDCG does, and less steeply than NDCG for a hit deep in the
    list. Note this dataset's evaluation protocol (new_datasets.py's
    NCFTestDatasetSampled: exactly one positive per user among the sampled
    candidates) means Recall@K is mathematically identical to HR@K here -
    with a single relevant item, "was it retrieved in the top K" (HR) and
    "what fraction of relevant items were retrieved" (Recall) are the same
    yes/no question. Adding a separately-named Recall@K metric on top of
    HR@K would just be reporting the same number twice under two names, so
    it isn't added here; MRR is the metric that's actually new information.
    """
    if gt_item in pred_items:
        idx = pred_items.index(gt_item)
        return 1.0 / (idx + 1)
    return 0.0


def evaluate_topk_pure(model, test_loader, top_k, device, return_mrr=False):
    HR, NDCG, MRR = [], [], []
    model.eval()
    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            preds = model(user, item)
            _, indices = torch.topk(preds, top_k)
            recommends = item[indices].cpu().numpy().tolist()
            gt_item = item[0].item()

            HR.append(hit_metric(gt_item, recommends))
            NDCG.append(ndcg_metric(gt_item, recommends))
            MRR.append(mrr_metric(gt_item, recommends))

    if return_mrr:
        return float(np.mean(HR)), float(np.mean(NDCG)), float(np.mean(MRR))
    return float(np.mean(HR)), float(np.mean(NDCG))


def evaluate_topk_isomap(ncf_model, isomap_model, test_loader, top_k, device, return_mrr=False):

    if isinstance(device, str):
        device = torch.device(device)

    ncf_model.eval()
    isomap_model.eval()

    HR, NDCG, MRR = [], [], []

    with torch.no_grad():
        item_Z = isomap_model().to(torch.float32)

    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)

            preds = ncf_model(user, item, item_Z)

            _, indices = torch.topk(preds, top_k)
            recommends = item[indices].cpu().numpy().tolist()

            gt_item = item[0].item()

            HR.append(hit_metric(gt_item, recommends))
            NDCG.append(ndcg_metric(gt_item, recommends))
            MRR.append(mrr_metric(gt_item, recommends))

    hr_mean = float(np.mean(HR)) if HR else 0.0
    ndcg_mean = float(np.mean(NDCG)) if NDCG else 0.0
    mrr_mean = float(np.mean(MRR)) if MRR else 0.0
    if return_mrr:
        return hr_mean, ndcg_mean, mrr_mean
    return hr_mean, ndcg_mean
